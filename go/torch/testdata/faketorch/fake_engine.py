"""Stand-in for the OA PyTorch engine: the real engine script's protocol,
containment and failure handling, with model sets that need no torch.

modelhost's tests run it as the engine script. --real names the real
oa_torch_engine.py, whose Session, Wire and jobs this runs; --mode picks a
behaviour, and --image-mode one for the image set alone; --record names a
file the stand-in writes its loads, unloads and jobs to.
"""
import importlib.util
import json
import os
import sys
import time

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
spec = importlib.util.spec_from_file_location("oa_torch_engine", ARGS["real"])
real = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real)
MODE = ARGS.get("mode", "ok")
IMAGE_MODE = ARGS.get("image-mode", "ok")
STEP = float(ARGS.get("step", "0.01"))
STEPS = int(ARGS.get("steps", "3"))


def record(**kw):
    path = ARGS.get("record")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(kw) + "\n")


class FakeRuntime(real.TorchRuntime):
    """The real runtime's checks, classification and phase hook, with no
    torch."""

    def prepare(self, msg, stage):
        fixes = msg.get("fixes")
        if not isinstance(fixes, list) or set(fixes) != real.KNOWN_FIXES:
            raise real.EngineFailure("config")
        real.add_site_packages(msg)

    def device_lost(self, exc):
        return real.device_lost_marker(exc)

    def out_of_memory(self, exc):
        return isinstance(exc, MemoryError)

    def after_cancel(self):
        record(event="after_cancel")

    def release(self):
        record(event="release")


class FakeSet:
    def __init__(self, name, kind, mode, stages):
        self.name, self.kind, self.mode, self.stages = name, kind, mode, stages

    def load(self, msg, stage, runtime):
        path = list(sys.path)
        runtime.prepare(msg, stage)
        # The text encode is wrapped as the real sets wrap theirs.
        self.encode = runtime.phased("text_encode", lambda: None)
        record(event="load", set=self.name, load=msg, env=dict(os.environ), argv=sys.argv, cwd=os.getcwd(), path=path,
               path_after=list(sys.path),
               flags={"isolated": sys.flags.isolated, "dont_write_bytecode": sys.flags.dont_write_bytecode,
                      "no_site": sys.flags.no_site})
        if msg.get("set") != self.name:
            raise real.EngineFailure("config")
        # Every model file through the real receiver: a lent handle in the
        # sandbox, the held path outside it; its size and SHA-256 are the
        # pin's.
        for role, entry in msg["files"].items():
            real.entry_bytes(entry)
        for module, entry in msg["source"].items():
            if real.sha256_file(entry["path"]) != entry["sha256"]:
                raise real.EngineFailure("source")
        for table, entry in msg["tables"].items():
            if real.sha256_file(entry["path"]) != entry["sha256"]:
                raise real.EngineFailure("files")
        stage(self.stages[0])
        if self.mode == "error-load":
            raise real.EngineFailure("files")
        if self.mode == "oom-load":
            raise MemoryError()
        if self.mode == "device-load":
            raise RuntimeError("HIP error: an illegal memory access was encountered")
        if self.mode == "crash-load":
            os._exit(5)
        if self.mode == "never-ready":
            time.sleep(3600)
        if self.mode == "slow-load":
            time.sleep(float(ARGS.get("load-seconds", "2")))
        if self.mode == "garbage":
            os.write(real_stdout_fd(), b"this is not a header\n")
            time.sleep(3600)
        stage(self.stages[1])

    def unload(self):
        record(event="unload", set=self.name)
        if self.mode == "hang-unload":
            time.sleep(3600)

    def generate(self, job, report):
        record(event="generate", set=self.name, id=job.id, prompt_bytes=len(job.prompt.encode()), seed=job.seed,
               width=job.width, height=job.height, frames=job.frames, fps=getattr(job, "fps", None),
               first_pixel=list(getattr(job, "frame", b"")[:3]))
        self.encode()
        for i in range(1, STEPS + 1):
            time.sleep(STEP)
            report("step", i, STEPS)
            if self.mode == "device-line-alive" and i == 1:
                # A device-lost line in the engine's own form while the engine
                # keeps answering: not a reset.
                print("oa-torch-engine: device-lost: hipErrorLaunchFailure", file=sys.stderr, flush=True)
        if self.mode == "crash-generate":
            os._exit(7)
        if self.mode == "fault-generate":
            native_fault()
        if self.mode == "device-lost":
            raise RuntimeError("HIP error: unspecified launch failure\nCompile with TORCH_USE_HIP_DSA")
        if self.mode == "oom":
            raise MemoryError()
        if self.mode == "fail":
            raise ValueError("an engine-side failure whose text never leaves the engine")
        if self.kind == "image":
            report("decode", 1, 1)
            pixels = bytes(((x * 7 + y * 3 + c) % 256) for y in range(job.height) for x in range(job.width) for c in range(3))
            return [pixels], int(ARGS.get("peak", "1000"))
        for i in (1, 2):
            report("decode", i, 2)
        frames = [bytes([(40 + i) % 256]) * (job.width * job.height * 3) for i in range(job.frames)]
        return frames, int(ARGS.get("peak", "1000"))


def native_fault():
    """A real access violation outside any handler, as a fault in a native
    library would be. ctypes guards its own calls on Windows, so the fault
    is a thread started at address zero; elsewhere a read of address zero."""
    import ctypes
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateThread.restype = ctypes.c_void_p
        kernel32.CreateThread(None, 0, None, None, 0, None)
        time.sleep(30)
    else:
        ctypes.string_at(0)
    os._exit(9)


def real_stdout_fd():
    return real.GARBAGE_FD


if ARGS.get("no-hello") == "1":
    time.sleep(3600)
    os._exit(0)


if MODE == "garbage" or IMAGE_MODE == "garbage":
    # The protocol's own stdout descriptor, to break the framing on purpose.
    original = real._claim_stdio

    def claim():
        r, w = original()
        real.GARBAGE_FD = w.fileno()
        return r, w

    real._claim_stdio = claim

real.serve({"hummingbird-xt-i2v": FakeSet("hummingbird-xt-i2v", "video", MODE, ("transformer", "turbovaed")),
            "flux2-klein-4b": FakeSet("flux2-klein-4b", "image", IMAGE_MODE, ("flux-transformer", "qwen3"))},
           FakeRuntime())
