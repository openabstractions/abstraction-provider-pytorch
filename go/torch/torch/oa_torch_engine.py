"""OpenAbstractions PyTorch engine: model sets served over stdio.

modelhost starts this script with the reviewed PyTorch-on-ROCm stack and
speaks to it over the process's own stdin and stdout; the engine opens no
socket and no port (CONTRACT.md MODELHOST-T1..T11). stderr is the engine log,
kept by the provider and never forwarded to a caller. One process serves
Hummingbird-XT video and FLUX.2 [klein] images, either set or both: the
first load names the set the process starts with, and a later load adds
the other beside it.

Framing, both directions: one header line, a JSON object of at most 64 KiB
ending in a line feed; a header with "bytes": N is followed by exactly N raw
bytes. The provider sends load, unload, generate, cancel and ping; the
engine answers hello, loading, ready, error, unloaded, progress, frame,
done, failed, cancelled and pong.

The interpreter runs with -I -S -B, so no site module, no .pth file and no
site-packages folder is on the path at start. The script imports the
standard library at start. At the first load it appends the venv's
site-packages the provider names, then imports torch and safetensors; a
video set imports AMD's reviewed Hummingbird-XT modules at the paths the
provider proved and holds, and an image set the venv's diffusers and
transformers; those modules import their own dependencies (einops, timm).
Model files load as safetensors only (MODELBRIDGE-L20) and torch.load is
replaced by a refusal. A Python audit hook refuses every non-loopback
socket operation before anything else is imported. The network-capable
packages diffusers and transformers import, the Hugging Face hub client and
its HTTP stack, are stub modules whose every call raises OANetworkDenied;
they need not be installed, and tokenizers load from their pinned files'
bytes alone (load_tokenizer). On Windows the provider starts this engine
inside an AppContainer with no network capability, which is its network
containment (PYTORCH-T12): native code and child processes reach no TCP,
DNS or UDP either. There the engine may not open a model file by path; the
provider lends each as a read-only handle, an fd: URI the receiver maps
(model_bytes, entry_bytes).
"""
import sys

# ---- network containment: installed before any other import ------------
import socket

LOOPBACK = {"127.0.0.1", "::1", "localhost"}
REFUSALS = []


def _host(addr):
    if isinstance(addr, tuple) and addr:
        addr = addr[0]
    if isinstance(addr, bytes):
        addr = addr.decode("ascii", "replace")
    return addr if isinstance(addr, str) or addr is None else repr(addr)


def _audit(event, args):
    """Refuse every urllib request and http.client connection, loopback
    included, since on Windows urllib takes the system proxy from the
    registry and a proxy on 127.0.0.1 would carry the request out; the
    engine uses neither. Refuse every non-loopback socket operation Python
    can audit: connect and connect_ex, bind, sendto, sendmsg and the name
    lookups asyncio's create_connection and create_server call first. Audit
    hooks cannot be removed once added
    (research/security-reviews/pytorch-rocm-windows-2.11.0-rocm7.13.0.md)."""
    if event in ("urllib.Request", "http.client.connect"):
        REFUSALS.append(event)
        raise PermissionError(f"oa-torch-engine: {event} refused")
    if not event.startswith("socket."):
        return
    if event in ("socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg"):
        sock = args[0] if args else None
        if getattr(sock, "family", None) == getattr(socket, "AF_UNIX", -1):
            return
        host = _host(args[1]) if len(args) > 1 else None
    elif event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex",
                   "socket.gethostbyaddr", "socket.getnameinfo"):
        host = _host(args[0]) if args else None
    else:
        return
    if host is None or host in LOOPBACK:
        return
    REFUSALS.append(event)
    raise PermissionError(f"oa-torch-engine: {event} refused")


def _refuse_listen():
    """Refuse listen on anything but a loopback-bound socket."""
    real_listen = socket.socket.listen

    def listen(self, *a, **k):
        try:
            host = self.getsockname()[0]
        except OSError:
            host = None
        if self.family == getattr(socket, "AF_UNIX", -1) or host in LOOPBACK:
            return real_listen(self, *a, **k)
        REFUSALS.append("socket.listen")
        raise PermissionError("oa-torch-engine: listen refused")

    socket.socket.listen = listen


def _refuse_asyncio_iocp():
    """asyncio's Windows proactor connects and sends through _overlapped,
    which raises no audit event; check the address at the proactor."""
    if sys.platform != "win32":
        return
    from asyncio import windows_events

    proactor = windows_events.IocpProactor

    def guard(name, real):
        def method(self, conn, *a, **k):
            addr = a[0] if name == "connect" else k.get("addr", a[2] if len(a) > 2 else None)
            host = _host(addr)
            if host is not None and host not in LOOPBACK and getattr(conn, "family", None) != getattr(socket, "AF_UNIX", -1):
                REFUSALS.append("asyncio." + name)
                raise PermissionError(f"oa-torch-engine: asyncio {name} refused")
            return real(self, conn, *a, **k)
        return method

    proactor.connect = guard("connect", proactor.connect)
    proactor.sendto = guard("sendto", proactor.sendto)


sys.addaudithook(_audit)
_refuse_listen()
_refuse_asyncio_iocp()

# ---- crash trace: a native fault prints every thread's Python stack -----
# to stderr, the engine log, before the process ends with its status; the
# provider records the status itself (MODELHOST-E1). Enabled before torch
# or any extension module is imported.
import faulthandler  # noqa: E402

faulthandler.enable(file=sys.stderr, all_threads=True)

import gc  # noqa: E402
import hashlib  # noqa: E402
import importlib.abc  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import queue  # noqa: E402
import re  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import types  # noqa: E402


# ---- network-capable packages: stubs, never their code -----------------
# Diffusers and transformers import the Hugging Face hub client and its HTTP
# stack (huggingface_hub, requests, urllib3, httpx and the rest) at import
# time, though this engine loads every file from a path the provider proved.
# A finder at the front of sys.meta_path answers each of those packages with
# a stub module tree: the import succeeds, the constants and decorators the
# importers read at import time are given below, every other name is a
# class whose call raises OANetworkDenied naming it, and importlib.metadata
# reports the pinned version a dependency check reads. The packages need not
# be installed at all (research/security-reviews/pytorch-rocm-windows-2.11.0-rocm7.13.0.md,
# addendum "Lean dependencies"). Native code in torch and ROCm is outside
# Python's reach: the provider's OS-level containment holds that.
import importlib.machinery  # noqa: E402
import importlib.metadata  # noqa: E402


class OANetworkDenied(PermissionError):
    """A call into a network-capable package this engine replaces with a stub."""


# Import names answered by stubs, and the distributions whose version a
# dependency check reads (transformers' dependency_versions_check, diffusers'
# and transformers' is_*_available).
DENIED_MODULES = ("huggingface_hub", "requests", "urllib3", "httpx", "httpcore", "h11", "anyio", "aiohttp",
                  "fsspec", "certifi", "charset_normalizer", "idna")
DENIED_DISTRIBUTIONS = {"huggingface-hub": "0.36.2", "requests": "2.34.2", "urllib3": "2.8.0", "httpx": "0.28.1",
                        "httpcore": "1.0.9", "h11": "0.16.0", "anyio": "4.15.1", "fsspec": "2026.9.0",
                        "certifi": "2026.7.22", "charset-normalizer": "3.5.2", "idna": "3.20"}
# Distributions a dependency check names that only the hub client imported:
# their version is reported and their modules are absent, so an import of
# one fails as any missing module does.
VERSION_ONLY_DISTRIBUTIONS = {"filelock": "4.0.8"}
_IDENTITY = object()
_HUB_HOME = os.environ.get("HF_HOME") or "oa-no-hub-home"
# What an importer reads from a stub at import time and gets as a value:
# versions, constants and decorators, from the import trace of every set's
# load (addendum "Lean dependencies"). No URL names a reachable host.
STUB_VALUES = {
    "huggingface_hub": {"__version__": DENIED_DISTRIBUTIONS["huggingface-hub"]},
    "huggingface_hub.constants": {
        "HF_HOME": _HUB_HOME, "HF_HUB_CACHE": os.path.join(_HUB_HOME, "hub"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(_HUB_HOME, "hub"), "default_cache_path": os.path.join(_HUB_HOME, "hub"),
        "HF_ASSETS_CACHE": os.path.join(_HUB_HOME, "assets"),
        "HUGGINGFACE_ASSETS_CACHE": os.path.join(_HUB_HOME, "assets"), "HF_HUB_OFFLINE": True,
        "HF_HUB_DISABLE_TELEMETRY": True, "HF_HUB_DISABLE_IMPLICIT_TOKEN": True, "HF_HUB_DISABLE_PROGRESS_BARS": True,
        "ENDPOINT": "oa-no-network:", "HUGGINGFACE_CO_URL_HOME": "oa-no-network:/",
        "HUGGINGFACE_CO_URL_TEMPLATE": "oa-no-network:/{repo_id}/{revision}/{filename}", "DEFAULT_REVISION": "main",
        "CONFIG_NAME": "config.json", "REPO_TYPE_MODEL": "model", "REPO_TYPE_DATASET": "dataset",
        "REPO_TYPE_SPACE": "space", "REPO_TYPES": ["model", "dataset", "space"],
        "SAFETENSORS_SINGLE_FILE": "model.safetensors", "SAFETENSORS_INDEX_FILE": "model.safetensors.index.json",
        "SAFETENSORS_WEIGHTS_FILE_PATTERN": "model{suffix}.safetensors",
        "PYTORCH_WEIGHTS_NAME": "pytorch_model.bin", "PYTORCH_WEIGHTS_FILE_PATTERN": "pytorch_model{suffix}.bin",
        "TF2_WEIGHTS_NAME": "tf_model.h5", "TF_WEIGHTS_NAME": "model.ckpt", "FLAX_WEIGHTS_NAME": "flax_model.msgpack",
        "REPOCARD_NAME": "README.md", "DEFAULT_DOWNLOAD_TIMEOUT": 10, "DEFAULT_ETAG_TIMEOUT": 10,
        "DEFAULT_REQUEST_TIMEOUT": 10, "HF_HUB_DOWNLOAD_TIMEOUT": 10, "HF_HUB_ETAG_TIMEOUT": 10,
        "HF_HUB_ENABLE_HF_TRANSFER": False, "HF_HUB_DISABLE_XET": True, "HF_DEBUG": False,
        "ENV_VARS_TRUE_VALUES": {"1", "ON", "YES", "TRUE"},
        "ENV_VARS_TRUE_AND_AUTO_VALUES": {"1", "ON", "YES", "TRUE", "AUTO"},
    },
    "huggingface_hub.utils": {"validate_hf_hub_args": _IDENTITY, "are_progress_bars_disabled": lambda *a, **k: True,
                              "is_jinja_available": lambda *a, **k: True},
}
# Every name read from a stub, with the module that read it first.
STUB_USES = {}


class _DeniedType(type):
    """A stub name: importers subclass it, annotate with it and test
    isinstance against it; calling it, or anything made from it, raises."""

    def __call__(cls, *a, **k):
        raise OANetworkDenied(f"oa-torch-engine: {cls.__qualname__} is a network call this engine refuses")


def _identity_decorator(fn=None, *a, **k):
    if callable(fn) and not a and not k:
        return fn
    return lambda f: f


def _reader():
    """The module whose code read a stub name: the first frame outside the
    import machinery, the stubs and this engine."""
    f = sys._getframe(2)
    while f is not None:
        name = f.f_globals.get("__name__", "")
        internal = name.startswith(("importlib", "_frozen_importlib")) or name.split(".")[0] in DENIED_MODULES
        if name and not internal and name != __name__:
            return name
        f = f.f_back
    return None


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        values = STUB_VALUES.get(self.__name__, {})
        if name in values:
            value = values[name]
        elif name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        else:
            value = _DeniedType(name, (Exception,), {"__module__": self.__name__,
                                                    "__qualname__": self.__name__ + "." + name})
        if value is _IDENTITY:
            value = _identity_decorator
        STUB_USES.setdefault(self.__name__ + "." + name, _reader())
        setattr(self, name, value)
        return value


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        m = _StubModule(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        module.__file__ = "<oa network stub " + module.__name__ + ">"


class _StubDistribution(importlib.metadata.Distribution):
    def __init__(self, name, version):
        self._text = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"

    def read_text(self, filename):
        return self._text if filename in ("METADATA", "PKG-INFO") else None

    def locate_file(self, path):
        return path


class _DenyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] not in DENIED_MODULES:
            return None
        return importlib.machinery.ModuleSpec(fullname, _StubLoader(), is_package=True)

    def find_distributions(self, context=importlib.metadata.DistributionFinder.Context()):
        known = {**DENIED_DISTRIBUTIONS, **VERSION_ONLY_DISTRIBUTIONS}
        name = context.name
        if name is None:
            return iter([_StubDistribution(n, v) for n, v in known.items()])
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key in known:
            return iter([_StubDistribution(key, known[key])])
        return iter(())


sys.meta_path.insert(0, _DenyFinder())

PROTOCOL = 2
MAX_HEADER = 64 << 10
MAX_PROMPT = 2048
MAX_FRAME_BYTES = 4096 * 4096 * 3
ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
T0 = time.monotonic()


def log(event, **kw):
    """One diagnostic line on stderr, the engine log. Never a prompt."""
    kw = {"t": round(time.monotonic() - T0, 3), "event": event, **kw}
    print("oa-torch-engine: " + json.dumps(kw), file=sys.stderr, flush=True)


def settled(torch, report):
    """Progress that names work the device has finished. The GPU runs a
    step's kernels after the Python that queued them returns, so a step's
    callback runs while the step is still queued: the live walkthrough of
    2026-10-02 (D10) saw FLUX.2's four steps arrive within 0.1 s and a
    video's decode sit at 99 percent for 15 s. Each report first waits for
    the device, so steps and decode chunks arrive as they finish."""
    def send(phase, done, total):
        torch.cuda.synchronize()
        report(phase, done, total)
    return send


class ProtocolError(Exception):
    pass


class Cancelled(Exception):
    pass


class EngineFailure(Exception):
    """A failure the engine reports by code alone: no engine text reaches a caller."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


# Codes a failed or error frame carries; the provider refuses any other.
CODES = {"load", "files", "source", "config", "oom", "device", "input", "error"}


# ---- stdio: the protocol owns fd 0 and fd 1 -----------------------------
def _claim_stdio():
    """Keep the provider's pipes for the protocol and point fd 0 at the null
    device and fd 1 at stderr, so no library's print or native write can put
    a byte into the protocol stream."""
    proto_in = os.dup(0)
    proto_out = os.dup(1)
    if sys.platform == "win32":
        import msvcrt
        msvcrt.setmode(proto_in, os.O_BINARY)
        msvcrt.setmode(proto_out, os.O_BINARY)
    os.dup2(2, 1)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    sys.stdout = sys.stderr
    sys.stdin = open(os.devnull, "r", encoding="utf-8")
    return os.fdopen(proto_in, "rb", buffering=0), os.fdopen(proto_out, "wb", buffering=0)


class Wire:
    def __init__(self, rfile, wfile):
        self.r, self.w = rfile, wfile
        self.lock = threading.Lock()

    def send(self, header, payload=b""):
        if payload:
            header = dict(header, bytes=len(payload))
        line = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
        if len(line) > MAX_HEADER:
            raise ProtocolError("header too long")
        with self.lock:
            self.w.write(line)
            if payload:
                view = memoryview(payload)
                while view:
                    n = self.w.write(view)
                    view = view[n:]

    def _exact(self, n):
        out = bytearray()
        while len(out) < n:
            chunk = self.r.read(n - len(out))
            if not chunk:
                raise EOFError
            out += chunk
        return bytes(out)

    def receive(self):
        """One message: the header object and its payload. EOF ends the engine."""
        line = bytearray()
        while not line.endswith(b"\n"):
            b = self.r.read(1)
            if not b:
                raise EOFError
            line += b
            if len(line) > MAX_HEADER:
                raise ProtocolError("header too long")
        try:
            header = json.loads(bytes(line))
        except ValueError:
            raise ProtocolError("header is not JSON")
        if not isinstance(header, dict) or not isinstance(header.get("op"), str):
            raise ProtocolError("header has no op")
        n = header.get("bytes", 0)
        if not isinstance(n, int) or isinstance(n, bool) or n < 0 or n > MAX_FRAME_BYTES:
            raise ProtocolError("bad payload length")
        return header, self._exact(n) if n else b""


# ---- jobs ----------------------------------------------------------------
def _number(header, name, low, high):
    v = header.get(name)
    if not isinstance(v, int) or isinstance(v, bool) or v < low or v > high:
        raise EngineFailure("input")
    return v


def _job_id(header):
    job_id = header.get("id")
    if not isinstance(job_id, str) or not ID.match(job_id):
        raise ProtocolError("bad job id")
    return job_id


def _prompt(header):
    prompt = header.get("prompt")
    if not isinstance(prompt, str) or not prompt or len(prompt.encode("utf-8")) > MAX_PROMPT:
        raise EngineFailure("input")
    return prompt


class Job:
    """A video: the prompt, seed, size, frames and rate, and the first frame
    as raw RGB rows."""

    def __init__(self, header, payload):
        self.id = _job_id(header)
        self.prompt = _prompt(header)
        # video@1's seed bound (sdcpp.MaxSeed); torch.Generator takes 64 bits.
        self.seed = _number(header, "seed", 0, 2**53)
        self.width = _number(header, "width", 16, 4096)
        self.height = _number(header, "height", 16, 4096)
        self.frames = _number(header, "frames", 1, 1024)
        self.fps = _number(header, "fps", 1, 120)
        if (self.frames - 1) % 4 or self.width % 32 or self.height % 32:
            raise EngineFailure("input")
        if len(payload) != self.width * self.height * 3:
            raise EngineFailure("input")
        self.frame = payload
        self.cancel = threading.Event()

    def check(self):
        if self.cancel.is_set():
            raise Cancelled()


class ImageJob:
    """An image: the prompt, seed and size; no payload. Its one frame is the
    image's raw RGB rows."""

    def __init__(self, header, payload):
        self.id = _job_id(header)
        self.prompt = _prompt(header)
        self.seed = _number(header, "seed", 0, 2**53)
        self.width = _number(header, "width", 16, 4096)
        self.height = _number(header, "height", 16, 4096)
        if self.width % 16 or self.height % 16 or payload:
            raise EngineFailure("input")
        self.frames = 1
        self.cancel = threading.Event()

    def check(self):
        if self.cancel.is_set():
            raise Cancelled()


JOBS = {"video": Job, "image": ImageJob}


def _set_name(header):
    name = header.get("set")
    if not isinstance(name, str) or not name or len(name) > 64:
        raise ProtocolError("bad set name")
    return name


class Session:
    """The protocol loop over the model sets this engine serves, by name;
    runtime is what they share in the process. Tests give stand-ins."""

    def __init__(self, wire, sets, runtime):
        self.wire, self.sets, self.runtime = wire, sets, runtime
        self.loaded = set()
        self.work = queue.Queue()
        self.current = None
        self.mu = threading.Lock()
        self.healthy = True

    def read(self):
        """Reader thread: ping and cancel are answered at once, even while a
        job runs; load, unload and generate wait for the main thread."""
        try:
            while True:
                header, payload = self.wire.receive()
                op = header["op"]
                if op == "ping":
                    seq = header.get("seq")
                    if not isinstance(seq, int) or isinstance(seq, bool):
                        raise ProtocolError("bad ping")
                    self.wire.send({"type": "pong", "seq": seq, "healthy": self.healthy})
                elif op == "cancel":
                    with self.mu:
                        job = self.current
                    if job is not None and job.id == header.get("id"):
                        job.cancel.set()
                elif op in ("load", "unload", "generate"):
                    self.work.put((header, payload))
                else:
                    raise ProtocolError("unknown op")
        except EOFError:
            log("stdin_closed")
            os._exit(0)
        except ProtocolError as e:
            log("protocol_error", why=str(e))
            os._exit(2)

    def failed(self, job_id, exc):
        """Report a job's failure by code; a lost device ends the engine."""
        marker = self.runtime.device_lost(exc)
        if marker:
            self.healthy = False
            print(f"oa-torch-engine: device-lost: {marker}", file=sys.stderr, flush=True)
            self.wire.send({"type": "failed", "id": job_id, "code": "device"})
            os._exit(3)
        code = exc.code if isinstance(exc, EngineFailure) else ("oom" if self.runtime.out_of_memory(exc) else "error")
        if code == "error":
            traceback.print_exception(exc, file=sys.stderr)
        self.wire.send({"type": "failed", "id": job_id, "code": code})

    def load(self, header, first):
        """Load one set. The first set's failure ends the engine; a later
        set's leaves the loaded ones serving, unless the device was lost."""
        name = _set_name(header)
        model = self.sets.get(name)
        if name in self.loaded:
            self.wire.send({"type": "ready", "set": name})
            return
        def stage(name_of_stage):
            # Each load stage is a log line too, so a load that ends the
            # process can be placed in it (MODELHOST-E1).
            log("load", set=name, stage=name_of_stage)
            self.wire.send({"type": "loading", "set": name, "stage": name_of_stage})

        self.runtime.on_phase = None
        try:
            if model is None:
                raise EngineFailure("config")
            model.load(header, stage, self.runtime)
        except Exception as exc:
            marker = self.runtime.device_lost(exc)
            if marker:
                self.healthy = False
                print(f"oa-torch-engine: device-lost: {marker}", file=sys.stderr, flush=True)
            code = exc.code if isinstance(exc, EngineFailure) else ("device" if marker else ("oom" if self.runtime.out_of_memory(exc) else "load"))
            traceback.print_exception(exc, file=sys.stderr)
            self.wire.send({"type": "error", "set": name, "code": code})
            if first or marker:
                os._exit(3 if marker else 1)
            if model is not None:
                model.unload()
            self.runtime.release()
            return
        self.loaded.add(name)
        self.wire.send({"type": "ready", "set": name})

    def unload(self, header):
        name = _set_name(header)
        self.runtime.on_phase = None
        if name in self.loaded:
            log("unload", set=name)
            self.sets[name].unload()
            self.loaded.discard(name)
            self.runtime.release()
            log("unloaded", set=name)
        self.wire.send({"type": "unloaded", "set": name})

    def generate(self, header, payload):
        name = _set_name(header)
        model = self.sets.get(name)
        if model is None or name not in self.loaded:
            self.wire.send({"type": "failed", "id": _job_id(header), "code": "config"})
            return
        try:
            job = JOBS[model.kind](header, payload)
        except EngineFailure as e:
            self.wire.send({"type": "failed", "id": header.get("id"), "code": e.code})
            return
        with self.mu:
            self.current = job
        # The job's start, and the set's phases while it runs, are log lines
        # for every set, so a job that ends the process can be placed in it
        # (MODELHOST-E1). Never the prompt.
        log("generate", id=job.id, set=name, width=job.width, height=job.height, frames=job.frames)
        self.runtime.on_phase = lambda phase, edge: log(phase, id=job.id, at=edge)
        try:
            frames, peak = model.generate(job, self.progress(job))
            job.check()
            for i, data in enumerate(frames):
                job.check()
                self.wire.send({"type": "frame", "id": job.id, "index": i}, data)
            self.wire.send({"type": "done", "id": job.id, "frames": job.frames, "peak": peak})
        except Cancelled:
            self.runtime.after_cancel()
            self.wire.send({"type": "cancelled", "id": job.id})
        except Exception as exc:
            self.failed(job.id, exc)
        finally:
            self.runtime.on_phase = None
            with self.mu:
                self.current = None

    def run(self):
        self.wire.send({"type": "hello", "protocol": PROTOCOL, "pid": os.getpid()})
        threading.Thread(target=self.read, name="oa-protocol", daemon=True).start()
        header, _ = self.work.get()
        if header["op"] != "load":
            log("protocol_error", why="the first operation is not load")
            os._exit(2)
        try:
            self.load(header, first=True)
            while True:
                header, payload = self.work.get()
                op = header["op"]
                if op == "load":
                    self.load(header, first=False)
                elif op == "unload":
                    self.unload(header)
                else:
                    self.generate(header, payload)
        except ProtocolError as e:
            log("protocol_error", why=str(e))
            os._exit(2)

    def progress(self, job):
        def report(phase, done, total):
            job.check()
            self.wire.send({"type": "progress", "id": job.id, "phase": phase, "done": done, "total": total})
            job.check()
        return report


# ---- the PyTorch model set -----------------------------------------------
# Configurations of the pinned public files, held here as constants so no
# configuration file is read: amd/HummingbirdXT @ 4d3bd2e3
# hm_2.2_5B/config.json (git blob abb356de) and
# vae/wan22_v1_tiling_16_12/config.json (git blob 8bdaa600); AMD-AGI/HummingbirdXT
# @ 929e90a2 infer/config/wan2.2/wan_civitai_5b.yaml.
HM_TRANSFORMER = {"dim": 3072, "eps": 1e-06, "ffn_dim": 14336, "freq_dim": 256, "in_dim": 48, "model_type": "ti2v",
                  "num_heads": 24, "num_layers": 30, "out_dim": 48, "text_len": 512}
HM_TURBOVAED = {"_class_name": "AutoencoderKLTurboVAED", "_diffusers_version": "0.32.0.dev0",
                "decoder_block_out_channels": [32, 64, 256, 512],
                "decoder_causal": False, "in_channels": 3, "latent_channels": 48,
                "decoder_layers_per_block": [4, 3, 3, 3, 4], "out_channels": 3, "patch_size": 2, "patch_size_t": 1,
                "resnet_norm_eps": 1e-06, "scaling_factor": 1.0,
                "decoder_spatio_temporal_scaling": [True, True, True, False],
                "decoder_spatio_only": [True, False, False, False],
                "decoder_is_dw_conv": [False, False, False, True, True], "decoder_dw_kernel_size": 5,
                "aligned_feature_projection_mode": "conv-2layer",
                "aligned_feature_projection_dim": [[512, 1024], [512, 8192]], "aligned_blks_indices": [0, 1],
                "scale_factor_spatial": 16, "scale_factor_temporal": 4}
# The YAML's vae_kwargs and text_encoder_kwargs as written; _accepted keeps
# the names each constructor takes, as the research harness did.
WAN_VAE_KWARGS = {"vae_type": "AutoencoderKLWan3_8", "vae_subpath": "Wan2.2_VAE.pth", "temporal_compression_ratio": 4,
                  "spatial_compression_ratio": 16}
UMT5_KWARGS = {"text_encoder_subpath": "models_t5_umt5-xxl-enc-bf16.pth", "tokenizer_subpath": "google/umt5-xxl",
               "text_length": 512, "vocab": 256384, "dim": 4096, "dim_attn": 4096, "dim_ffn": 10240, "num_heads": 64,
               "num_layers": 24, "num_buckets": 32, "shared_pos": False, "dropout": 0.0}
# TurboVAED's temporal tiling, AMD's run_i2v.sh: --t_block_size 16 --t_stride 12.
DECODE_BLOCK, DECODE_STRIDE = 16, 12

# The videox_fun modules this engine runs, by module name; any other
# videox_fun module is refused, and the package __init__ files, which import
# unrelated models and optional patch packages, are never run.
SOURCE_MODULES = (
    "videox_fun.models.attention",
    "videox_fun.models.wan22_turbo",
    "videox_fun.models.wan_vae3_8",
    "videox_fun.models.wan_text_encoder",
    "videox_fun.utils.fm_solvers",
    "videox_fun.utils.fm_solvers_unipc",
    "videox_fun.pipeline.pipeline_wan2_2_ti2v",
    "autoencoder_kl_turbo_vaed_ours_wan22",
)
STUB_PACKAGES = ("videox_fun", "videox_fun.models", "videox_fun.utils", "videox_fun.pipeline", "videox_fun.dist")
FILE_ROLES = ("transformer", "turbovaed", "wan-vae", "umt5-1", "umt5-2", "umt5-3",
              "tokenizer", "tokenizer-config", "tokenizer-special", "tokenizer-spiece")
# Fixes from the resource kit's pytorch-rocm dispatch row this engine applies
# (gpufault.DispatchFor), by the name the provider sends.
KNOWN_FIXES = {"cudnn-off", "seq-len-from-clip"}
# Device-lost markers: HIP's own error names, and the text hipGetErrorString
# gives each where it is known, which PyTorch's "HIP error: ..." carries.
DEVICE_LOST = (
    ("hipErrorLaunchFailure", "unspecified launch failure"),
    ("hipErrorIllegalAddress", "an illegal memory access was encountered"),
    ("hipErrorLaunchTimeOut", "the launch timed out and was terminated"),
    ("hipErrorContextIsDestroyed", "context is destroyed"),
    ("hipErrorIllegalState", None),
)


def device_lost_marker(exc):
    """The HIP error name a lost device raised exc with, or ""; only a HIP
    error PyTorch raised counts, never caller text."""
    if not isinstance(exc, RuntimeError):
        return ""
    text = str(exc)
    if "HIP error" not in text and "hipError" not in text:
        return ""
    for name, words in DEVICE_LOST:
        if name in text or (words and words in text):
            return name
    return ""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _SourceOnly(importlib.abc.Loader):
    """Runs a pinned module from its source bytes alone: no cached bytecode
    beside the file is ever read, and none is written."""

    def __init__(self, path, digest):
        self.path, self.digest = path, digest

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        with open(self.path, "rb") as f:
            source = f.read()
        if hashlib.sha256(source).hexdigest() != self.digest:
            raise EngineFailure("source")
        exec(compile(source, self.path, "exec", dont_inherit=True), module.__dict__)


class _PinnedFinder(importlib.abc.MetaPathFinder):
    def __init__(self, modules):
        self.modules = modules

    def find_spec(self, fullname, path=None, target=None):
        entry = self.modules.get(fullname)
        if entry is None:
            if fullname.startswith("videox_fun."):
                raise ImportError(f"{fullname} is not a module this engine runs")
            return None
        return importlib.util.spec_from_file_location(fullname, entry["path"], loader=_SourceOnly(entry["path"], entry["sha256"]))


def add_site_packages(msg):
    """The interpreter runs with -I -S: no site module and no .pth file ran,
    so no site-packages is on the path. The venv's, which the provider names,
    is appended now, after the network hook; nothing in it runs at startup,
    and under -S sys.prefix stays the base interpreter's."""
    site = msg.get("site_packages")
    if not isinstance(site, str) or not os.path.isabs(site) or not os.path.isdir(site):
        raise EngineFailure("config")
    if site not in sys.path:
        sys.path.append(site)


class _Lent:
    """One model file the provider lends: an fd: URI (VISION 2026-10-01,
    "model bytes") naming a read-only handle the provider duplicated into
    this process, whose sandbox may not open the file by path
    (PYTORCH-T12). Opened, it is a private copy-on-write mapping of the
    whole file; the handle is closed once mapped, and the mapping when the
    with block ends."""

    def __init__(self, entry):
        uri, size = entry.get("uri"), entry.get("size")
        m = re.fullmatch(r"fd:(\d+)\?offset=0&length=(\d+)&sha256=([0-9a-f]{64})", uri if isinstance(uri, str) else "")
        if m is None or not isinstance(size, int) or int(m.group(2)) != size or m.group(3) != entry.get("sha256"):
            raise EngineFailure("files")
        self.handle, self.size, self.view = int(m.group(1)), size, None

    def __enter__(self):
        import mmap
        if os.name == "nt":
            import msvcrt
            fd = msvcrt.open_osfhandle(self.handle, os.O_RDONLY)
        else:
            fd = self.handle
        try:
            if os.fstat(fd).st_size != self.size or self.size == 0:
                raise EngineFailure("files")
            self.view = mmap.mmap(fd, self.size, access=mmap.ACCESS_COPY)
        finally:
            os.close(fd)
        return self.view

    def __exit__(self, *exc):
        if self.view is not None:
            self.view.close()


def entry_bytes(entry):
    """The bytes of one small pinned file, a tokenizer's or a
    configuration's: through its lent handle, or from the path the provider
    holds it at outside a sandbox; a size or SHA-256 other than the pin's
    fails the load."""
    if "uri" in entry:
        with _Lent(entry) as view:
            data = bytes(view)
    else:
        path = entry.get("path")
        if not isinstance(path, str) or not os.path.isabs(path):
            raise EngineFailure("files")
        with open(path, "rb") as f:
            data = f.read()
    if len(data) != entry.get("size") or hashlib.sha256(data).hexdigest() != entry.get("sha256"):
        raise EngineFailure("files")
    return data


# safetensors' dtypes by their torch names and item sizes.
SAFETENSORS_DTYPES = {"F64": ("float64", 8), "F32": ("float32", 4), "F16": ("float16", 2), "BF16": ("bfloat16", 2),
                      "I64": ("int64", 8), "I32": ("int32", 4), "I16": ("int16", 2), "I8": ("int8", 1),
                      "U8": ("uint8", 1), "BOOL": ("bool", 1), "F8_E4M3": ("float8_e4m3fn", 1),
                      "F8_E5M2": ("float8_e5m2", 1)}
# The largest safetensors header this engine reads.
SAFETENSORS_HEADER = 100 << 20


def safetensors_layout(view):
    """Each tensor of a safetensors file as (name, torch dtype name, shape,
    start, end), start and end absolute in the file, in file order. The
    header is the format's: eight bytes of little-endian length, then JSON;
    a header past the bound, an unknown dtype, a size that does not match
    its shape, or data that overlaps, leaves a gap or runs past the file
    fails the load."""
    if len(view) < 8:
        raise EngineFailure("files")
    n = int.from_bytes(view[:8], "little")
    if n > SAFETENSORS_HEADER or 8 + n > len(view):
        raise EngineFailure("files")
    try:
        header = json.loads(bytes(view[8:8 + n]))
    except ValueError:
        raise EngineFailure("files") from None
    if not isinstance(header, dict):
        raise EngineFailure("files")
    base, out = 8 + n, []
    for name, t in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(t, dict) or t.get("dtype") not in SAFETENSORS_DTYPES:
            raise EngineFailure("files")
        shape, offsets = t.get("shape"), t.get("data_offsets")
        if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
            raise EngineFailure("files")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(o, int) for o in offsets):
            raise EngineFailure("files")
        dtype, item = SAFETENSORS_DTYPES[t["dtype"]]
        count = 1
        for d in shape:
            count *= d
        begin, end = offsets
        if not 0 <= begin <= end or end - begin != count * item:
            raise EngineFailure("files")
        out.append((name, dtype, shape, base + begin, base + end))
    out.sort(key=lambda x: x[3])
    position = base
    for _, _, _, begin, end in out:
        if begin != position:
            raise EngineFailure("files")
        position = end
    if position != len(view):
        raise EngineFailure("files")
    return out


def model_bytes(entry, device):
    """The receiver of one model file's bytes (VISION 2026-10-01, "model
    bytes"). In the sandbox the provider lends the file it holds as a
    read-only handle, an fd: URI, and each tensor is read from the mapping
    and moved to device, a copy on the CPU, so nothing keeps the mapping
    once loaded; outside it, the path the provider holds the file at."""
    if "uri" not in entry:
        from safetensors.torch import load_file
        path = entry.get("path")
        if not isinstance(path, str) or not os.path.isabs(path):
            raise EngineFailure("files")
        if os.path.getsize(path) != entry.get("size"):
            raise EngineFailure("files")
        return load_file(path, device=device)
    import torch
    out = {}
    with _Lent(entry) as view:
        for name, dtype, shape, begin, end in safetensors_layout(view):
            kind = getattr(torch, dtype)
            if end == begin:
                out[name] = torch.empty(shape, dtype=kind, device=device)
                continue
            item = SAFETENSORS_DTYPES_ITEM[dtype]
            cpu = torch.frombuffer(view, dtype=kind, count=(end - begin) // item, offset=begin).reshape(shape)
            out[name] = cpu.clone() if device == "cpu" else cpu.to(device)
            del cpu
    return out


SAFETENSORS_DTYPES_ITEM = {name: item for name, item in SAFETENSORS_DTYPES.values()}


class TorchRuntime:
    """What the sets of one process share: the venv's site-packages and
    torch, imported once at the first load, with torch.load refused and the
    pytorch-rocm dispatch row's fixes applied; and how a failure is told."""

    def __init__(self):
        self.torch = None
        # on_phase hears the start and end of a set's phases (its text
        # encode, a first frame's encode) while a job runs; Session.generate
        # sets it for the job and clears it after, so it is None during a
        # load or an unload (MODELHOST-E1).
        self.on_phase = None

    def phased(self, name, call):
        """call, with the running job's phase log line before and after it.
        A set wraps its modules with it at load."""
        def wrapped(*a, **k):
            if self.on_phase is not None:
                self.on_phase(name, "start")
            out = call(*a, **k)
            if self.on_phase is not None:
                self.on_phase(name, "end")
            return out
        return wrapped

    def prepare(self, msg, stage):
        fixes = msg.get("fixes")
        if not isinstance(fixes, list) or set(fixes) != KNOWN_FIXES or len(fixes) != len(KNOWN_FIXES):
            raise EngineFailure("config")
        add_site_packages(msg)
        if self.torch is not None:
            return self.torch
        stage("torch")
        import torch

        def refused(*a, **k):
            raise PermissionError("oa-torch-engine: torch.load (pickle) is refused")

        torch.load = refused

        def denied(name):
            def call(*a, **k):
                raise OANetworkDenied(f"oa-torch-engine: torch.hub.{name} is a network call this engine refuses")
            return call

        # torch.hub downloads through the standard library, which the audit
        # hook already refuses; these name the call instead.
        for name in ("load_state_dict_from_url", "download_url_to_file", "load", "list", "help"):
            setattr(torch.hub, name, denied(name))
        log("torch", version=torch.__version__, hip=getattr(torch.version, "hip", None))
        # The pytorch-rocm dispatch row: MIOpen convolution off, so
        # convolutions run on hipBLASLt and rocBLAS GEMMs and nothing is
        # searched (research/live-runs/2026-10-01-rocm.md fault 1).
        torch.backends.cudnn.enabled = False
        self.torch = torch
        return torch

    # -- classification, safe before torch is imported --
    def device_lost(self, exc):
        return device_lost_marker(exc)

    def out_of_memory(self, exc):
        torch = self.torch
        return torch is not None and isinstance(exc, torch.OutOfMemoryError)

    def after_cancel(self):
        if self.torch is not None:
            self.torch.cuda.empty_cache()

    def release(self):
        """Give an unloaded set's memory back to the device."""
        gc.collect()
        if self.torch is not None:
            self.torch.cuda.empty_cache()


def empty_parameters(torch):
    """Builds a module with its parameters on the meta device, as
    accelerate's init_empty_weights does with buffers left in place; the
    state dict is then assigned."""

    class Empty:
        def __enter__(inner):
            inner.old = torch.nn.Module.register_parameter

            def register(module, name, param):
                inner.old(module, name, param)
                if param is not None:
                    cls = type(module._parameters[name])
                    kwargs = module._parameters[name].__dict__
                    kwargs["requires_grad"] = param.requires_grad
                    module._parameters[name] = cls(module._parameters[name].to("meta"), **kwargs)

            torch.nn.Module.register_parameter = register

        def __exit__(inner, *exc):
            torch.nn.Module.register_parameter = inner.old

    return Empty()


def _tokenizer_files(files, roles):
    """The pinned bytes of the tokenizer files of roles, by the file name
    each is pinned under. Nothing is read from a folder, so only pinned
    bytes shape the tokenizer; a name given twice fails the load."""
    named = {}
    for role in roles:
        entry = files[role]
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or name in named or os.path.basename(name) != name:
            raise EngineFailure("files")
        named[name] = entry_bytes(entry)
    return named


# Special-token and length settings a tokenizer_config.json names, which a
# set's tokenizer takes as written.
TOKENIZER_SETTINGS = ("model_max_length", "padding_side", "truncation_side", "clean_up_tokenization_spaces",
                      "chat_template", "split_special_tokens")
SPECIAL_TOKENS = ("bos_token", "eos_token", "unk_token", "pad_token", "sep_token", "cls_token", "mask_token")


def load_tokenizer(named):
    """A set's tokenizer from its pinned files' bytes alone, by file name:
    the tokenizers library's own Tokenizer read from tokenizer.json, with the
    special tokens, length settings and chat template its
    tokenizer_config.json and special_tokens_map.json name, wrapped in
    transformers' PreTrainedTokenizerFast for the pipelines. No class is
    looked up, no hub path is resolved and no folder is read (addendum "Lean
    dependencies")."""
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    def read(name):
        if name not in named:
            return {}
        try:
            value = json.loads(named[name].decode("utf-8"))
        except ValueError:
            raise EngineFailure("files") from None
        if not isinstance(value, dict):
            raise EngineFailure("files")
        return value

    config, special = read("tokenizer_config.json"), read("special_tokens_map.json")

    def content(token):
        return token.get("content") if isinstance(token, dict) else token

    kwargs = {}
    for key in SPECIAL_TOKENS:
        token = content(config.get(key, special.get(key)))
        if token is not None:
            kwargs[key] = token
    extra = config.get("additional_special_tokens", special.get("additional_special_tokens"))
    if extra:
        kwargs["additional_special_tokens"] = [content(t) for t in extra]
    for key in TOKENIZER_SETTINGS:
        if key in config:
            kwargs[key] = config[key]
    if "chat_template" not in kwargs and "chat_template.jinja" in named:
        kwargs["chat_template"] = named["chat_template.jinja"].decode("utf-8")
    if "tokenizer.json" not in named:
        raise EngineFailure("files")
    # The model inputs AutoTokenizer's classes name for both sets: no
    # token_type_ids.
    return PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(named["tokenizer.json"].decode("utf-8")),
                                   model_input_names=["input_ids", "attention_mask"], **kwargs)


def _check_tables(tables):
    for entry in tables.values():
        if not isinstance(entry, dict) or not os.path.isabs(entry.get("path", "")) or sha256_file(entry["path"]) != entry.get("sha256"):
            raise EngineFailure("files")


class HummingbirdXT:
    """AMD's Hummingbird-XT image to video with TurboVAED, as the research
    harness ran it (research/live-runs/2026-10-01-rocm.md), with the reviewed
    patches: seq_len from the clip, SDPA in place of flash_attn, and stub
    packages around videox_fun."""

    kind = "video"
    name = "hummingbird-xt-i2v"

    def __init__(self):
        self.torch = None
        self.seq = {"tokens": 0}
        self.on_step = None
        self.on_decode = None
        self.modules = None
        self.pipe = None
        self.tv = None

    # -- load --
    def load(self, msg, stage, runtime):
        if msg.get("set") != self.name:
            raise EngineFailure("config")
        files, source, tables = msg.get("files"), msg.get("source"), msg.get("tables")
        options, preset = msg.get("options"), msg.get("preset")
        if not isinstance(files, dict) or sorted(files) != sorted(FILE_ROLES):
            raise EngineFailure("files")
        if not isinstance(source, dict) or sorted(source) != sorted(SOURCE_MODULES):
            raise EngineFailure("source")
        if not isinstance(tables, dict) or sorted(tables) != ["umt5", "wan-vae"]:
            raise EngineFailure("files")
        if not isinstance(options, dict) or set(options) - {"decode_dtype"} or options.get("decode_dtype", "f32") not in ("f32", "bf16"):
            raise EngineFailure("config")
        if not isinstance(preset, dict):
            raise EngineFailure("config")
        self.preset = {"timesteps": [int(t) for t in preset["timesteps"]], "shift": float(preset["shift"]),
                       "guidance": float(preset["guidance"]), "boundary": float(preset["boundary"])}
        for entry in source.values():
            if not isinstance(entry, dict) or not os.path.isabs(entry.get("path", "")) or not isinstance(entry.get("sha256"), str):
                raise EngineFailure("source")
        tokenizer = _tokenizer_files(files, ("tokenizer", "tokenizer-config", "tokenizer-special", "tokenizer-spiece"))
        _check_tables(tables)

        torch = runtime.prepare(msg, stage)
        self.torch = torch

        stage("source")
        wan22, vae38, t5, pipemod, turbo = self._import(source)
        dev = torch.device("cuda")
        decode_dtype = torch.bfloat16 if options.get("decode_dtype") == "bf16" else torch.float32

        stage("transformer")
        with empty_parameters(torch):
            tr = wan22.Wan22Model(**_accepted(wan22.Wan22Model, HM_TRANSFORMER))
        tr.load_state_dict(model_bytes(files["transformer"], "cuda"), assign=True, strict=True)
        tr = tr.to(torch.bfloat16).eval()

        stage("wan-vae")
        vae = vae38.AutoencoderKLWan3_8(**_accepted(vae38.AutoencoderKLWan3_8, WAN_VAE_KWARGS))
        table = _table(tables["wan-vae"]["path"])
        sd = {"model." + table[k]: v for k, v in model_bytes(files["wan-vae"], "cuda").items() if k in table}
        missing, unexpected = vae.load_state_dict(sd, strict=False)
        del sd
        if missing or unexpected:
            raise EngineFailure("files")
        vae = vae.to(dev, torch.bfloat16).eval()
        # The first frame's encode is the pipeline's only Wan VAE call.
        vae.encode = runtime.phased("image_encode", vae.encode)

        stage("umt5")
        with empty_parameters(torch):
            te = t5.WanT5EncoderModel(**_accepted(t5.WanT5EncoderModel, UMT5_KWARGS))
        table = _table(tables["umt5"]["path"])
        sd = {}
        for role in ("umt5-1", "umt5-2", "umt5-3"):
            for k, v in model_bytes(files[role], "cuda").items():
                if k in table:
                    sd[table[k]] = v
        te.load_state_dict(sd, assign=True, strict=True)
        del sd
        te = te.to(torch.bfloat16).eval()
        te.forward = runtime.phased("text_encode", te.forward)

        stage("turbovaed")
        tv = turbo.AutoencoderKLTurboVAED.from_config(config=dict(HM_TURBOVAED))
        ck = model_bytes(files["turbovaed"], "cpu")
        decoder_keys = tv.decoder.state_dict()
        tv.decoder.load_state_dict({k: v for k, v in ck.items() if k in decoder_keys})
        tv = tv.to(dev, decode_dtype).eval()
        tv.latent_channels = tv.config.latent_channels
        tv.spacial_compression_ratio = tv.config.scale_factor_spatial
        tv.temporal_compression_ratio = tv.config.scale_factor_temporal
        tv.config.temporal_compression_ratio = tv.config.scale_factor_temporal
        tv.use_framewise_decoding = True
        tv.tile_sample_min_num_frames = DECODE_BLOCK
        tv.tile_sample_stride_num_frames = DECODE_STRIDE
        decoder_forward = tv.decoder.forward

        def decoder_step(*a, **k):
            if self.on_decode is not None:
                self.on_decode(before=True)
            out = decoder_forward(*a, **k)
            if self.on_decode is not None:
                self.on_decode(before=False)
            return out

        tv.decoder.forward = decoder_step

        stage("pipeline")
        tok = load_tokenizer(tokenizer)
        sch = pipemod.FlowShiftScheduler(shift=self.preset["shift"], sigma_min=0.0, extra_one_step=True)
        self.pipe = pipemod.Wan2_2TI2VPipeline(transformer=tr, transformer_2=None, vae=vae, tokenizer=tok,
                                               text_encoder=te, scheduler=sch).to(device=dev)
        self.tv, self.turbo = tv, turbo
        torch.cuda.synchronize()
        log("loaded", set=self.name, gpu_reserved=torch.cuda.memory_reserved(), gpu_allocated=torch.cuda.memory_allocated())

    def unload(self):
        self.pipe, self.tv = None, None

    def _import(self, source):
        # AMD's modules are imported and patched once per process; a set
        # loaded again reuses them.
        if self.modules is not None:
            return self.modules
        root = os.path.dirname(os.path.abspath(__file__))
        for pkg in STUB_PACKAGES:
            m = types.ModuleType(pkg)
            m.__path__ = []
            m.__file__ = os.path.join(root, "<stub " + pkg + ">")
            sys.modules[pkg] = m
        sys.meta_path.insert(0, _PinnedFinder(source))
        models = sys.modules["videox_fun.models"]
        # AMD's videox_fun.models package exports transformers' AutoTokenizer,
        # which the pipeline module imports from there and never calls: the
        # engine gives the pipeline its tokenizer (load_tokenizer).
        models.AutoTokenizer = importlib.import_module("transformers").PreTrainedTokenizerFast
        wan22 = importlib.import_module("videox_fun.models.wan22_turbo")
        vae38 = importlib.import_module("videox_fun.models.wan_vae3_8")
        t5 = importlib.import_module("videox_fun.models.wan_text_encoder")
        for name in ("Wan22Model", "WanDiffusionWrapper"):
            setattr(models, name, getattr(wan22, name))
        models.AutoencoderKLWan3_8 = vae38.AutoencoderKLWan3_8
        models.WanT5EncoderModel = t5.WanT5EncoderModel
        for name in ("AutoencoderKLWan", "CLIPModel", "Wan2_2Transformer3DModel"):
            setattr(models, name, None)
        pipemod = importlib.import_module("videox_fun.pipeline.pipeline_wan2_2_ti2v")
        turbo = importlib.import_module("autoencoder_kl_turbo_vaed_ours_wan22")
        # flash_attn does not exist for Windows ROCm: PyTorch SDPA, honouring
        # k_lens (AOTriton's kernels through the dispatch row's variable).
        wan22.flash_attention = _sdpa_attention(self.torch)
        # seq_len from the clip, never AMD's constant 27280 (1280x704x121):
        # the padding multiplied the work about five times and, on the SDPA
        # math path, hung the GPU (research/live-runs/2026-10-01-rocm.md
        # fault 2; AMD-AGI/HummingbirdXT#3).
        wrapper = wan22.WanDiffusionWrapper
        init, forward = wrapper.__init__, wrapper.forward
        seq = self.seq

        def patched_init(this, *a, **k):
            init(this, *a, **k)
            this.seq_len = seq["tokens"]

        def patched_forward(this, *a, **k):
            this.seq_len = seq["tokens"]
            if self.on_step is not None:
                self.on_step(before=True)
            out = forward(this, *a, **k)
            if self.on_step is not None:
                self.on_step(before=False)
            return out

        wrapper.__init__, wrapper.forward = patched_init, patched_forward
        self.modules = (wan22, vae38, t5, pipemod, turbo)
        return self.modules

    # -- generate --
    def generate(self, job, report):
        torch = self.torch
        report = settled(torch, report)
        timesteps = self.preset["timesteps"]
        latent_frames = (job.frames - 1) // 4 + 1
        self.seq["tokens"] = latent_frames * (job.height // 32) * (job.width // 32)
        decode_total = _decode_chunks(latent_frames)
        state = {"step": 0, "decode": 0}
        # Session.generate logged the job's start; one log line per step and
        # decode chunk places a job that ends the process (MODELHOST-E1).
        # Never the prompt.
        log("plan", id=job.id, steps=len(timesteps), decode_chunks=decode_total, tokens=self.seq["tokens"])

        def on_step(before):
            if before:
                job.check()
                log("step", id=job.id, step=state["step"] + 1, at="start")
                return
            state["step"] += 1
            # The end line follows the wait for the device, so a fault in
            # the step ends the log before it.
            torch.cuda.synchronize()
            log("step", id=job.id, step=state["step"], at="end")
            report("step", min(state["step"], len(timesteps)), len(timesteps))

        def on_decode(before):
            if before:
                job.check()
                log("decode_chunk", id=job.id, chunk=state["decode"] + 1, at="start")
                return
            state["decode"] += 1
            torch.cuda.synchronize()
            log("decode_chunk", id=job.id, chunk=state["decode"], at="end")
            report("decode", min(state["decode"], decode_total), decode_total)

        self.on_step, self.on_decode = on_step, on_decode
        try:
            torch.cuda.reset_peak_memory_stats()
            img = torch.frombuffer(bytearray(job.frame), dtype=torch.uint8).view(job.height, job.width, 3)
            img = img.permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)
            img = img.to("cuda").unsqueeze(1).to(dtype=torch.bfloat16)
            g = torch.Generator(device="cpu").manual_seed(job.seed)
            with torch.no_grad():
                lat = self.pipe("t2v", job.prompt, num_frames=job.frames, negative_prompt=None, height=job.height,
                                width=job.width, generator=g, guidance_scale=self.preset["guidance"],
                                num_inference_steps=len(timesteps), boundary=self.preset["boundary"], img=img,
                                mask_video=None, shift=self.preset["shift"], input_timesteps=timesteps,
                                output_type="latent", seed=job.seed).videos
                job.check()
                log("decode", id=job.id, at="start", gpu_reserved=torch.cuda.memory_reserved())
                video = self.turbo.ours_decode(self.pipe, self.tv, lat)
            frames = (video[0].permute(1, 2, 3, 0).float() * 255).round().clamp(0, 255).to(torch.uint8).cpu()
            torch.cuda.synchronize()
            log("decode", id=job.id, at="end")
            peak = int(torch.cuda.max_memory_reserved())
            log("clip", tokens=self.seq["tokens"], frames=int(frames.shape[0]), gpu_peak_reserved=peak)
            if frames.shape[0] != job.frames or frames.shape[1] != job.height or frames.shape[2] != job.width:
                raise EngineFailure("error")
            return (frames[i].contiguous().numpy().tobytes() for i in range(job.frames)), peak
        finally:
            self.on_step, self.on_decode = None, None


# ---- FLUX.2 [klein] --------------------------------------------------------
# Configurations of black-forest-labs/FLUX.2-klein-4B @ e7b7dc27, held here
# as constants so no configuration file is read: transformer/config.json
# (git blob f21592ae), vae/config.json (c3f38eb4), scheduler/
# scheduler_config.json (7e53605e) and text_encoder/config.json (86f951d1),
# each without the keys naming its class, its version or the folder it was
# written from.
FLUX_TRANSFORMER = {"attention_head_dim": 128, "axes_dims_rope": [32, 32, 32, 32], "eps": 1e-06, "guidance_embeds": False,
                    "in_channels": 128, "joint_attention_dim": 7680, "mlp_ratio": 3.0, "num_attention_heads": 24,
                    "num_layers": 5, "num_single_layers": 20, "out_channels": None, "patch_size": 1, "rope_theta": 2000,
                    "timestep_guidance_channels": 256}
FLUX_VAE = {"act_fn": "silu", "batch_norm_eps": 0.0001, "batch_norm_momentum": 0.1, "block_out_channels": [128, 256, 512, 512],
            "down_block_types": ["DownEncoderBlock2D"] * 4, "force_upcast": True, "in_channels": 3, "latent_channels": 32,
            "layers_per_block": 2, "mid_block_add_attention": True, "norm_num_groups": 32, "out_channels": 3,
            "patch_size": [2, 2], "sample_size": 1024, "up_block_types": ["UpDecoderBlock2D"] * 4,
            "use_post_quant_conv": True, "use_quant_conv": True}
FLUX_SCHEDULER = {"base_image_seq_len": 256, "base_shift": 0.5, "invert_sigmas": False, "max_image_seq_len": 4096,
                  "max_shift": 1.15, "num_train_timesteps": 1000, "shift": 3.0, "shift_terminal": None,
                  "stochastic_sampling": False, "time_shift_type": "exponential", "use_beta_sigmas": False,
                  "use_dynamic_shifting": True, "use_exponential_sigmas": False, "use_karras_sigmas": False}
# The Qwen3 4B text encoder's configuration as written, which the research
# harness passed whole to Qwen3Config; it names no attention implementation
# and no auto_map, so no configuration-driven code loads (transformers
# CVE-2026-4372).
QWEN3 = {"architectures": ["Qwen3ForCausalLM"], "attention_bias": False, "attention_dropout": 0.0, "bos_token_id": 151643,
         "dtype": "bfloat16", "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu", "hidden_size": 2560,
         "initializer_range": 0.02, "intermediate_size": 9728, "layer_types": ["full_attention"] * 36,
         "max_position_embeddings": 40960, "max_window_layers": 36, "model_type": "qwen3", "num_attention_heads": 32,
         "num_hidden_layers": 36, "num_key_value_heads": 8, "rms_norm_eps": 1e-06, "rope_scaling": None,
         "rope_theta": 1000000, "sliding_window": None, "tie_word_embeddings": True, "transformers_version": "4.56.1",
         "use_cache": True, "use_sliding_window": False, "vocab_size": 151936}
FLUX_FILE_ROLES = ("flux-transformer", "flux-vae", "qwen3", "qwen3-tokenizer", "qwen3-tokenizer-config",
                   "qwen3-tokenizer-special", "qwen3-tokenizer-added", "qwen3-chat-template", "qwen3-vocab", "qwen3-merges")
FLUX_TOKENIZER_ROLES = ("qwen3-tokenizer", "qwen3-tokenizer-config", "qwen3-tokenizer-special", "qwen3-tokenizer-added",
                        "qwen3-chat-template", "qwen3-vocab", "qwen3-merges")


class Flux2Klein:
    """Black Forest Labs' FLUX.2 [klein] 4B text to image, as the research
    harness ran it (research/live-runs/2026-10-02.md), from Comfy-Org's single
    files: the transformer through Diffusers' own FLUX.2 key conversion, the
    VAE through the provider's key table, and the Qwen3 encoder by name with
    its tied output head."""

    kind = "image"
    name = "flux2-klein-4b"

    def __init__(self):
        self.torch = None
        self.pipe = None
        self.steps = 0
        self.guidance = 1.0
        self.on_decode = None

    def load(self, msg, stage, runtime):
        if msg.get("set") != self.name:
            raise EngineFailure("config")
        files, source, tables = msg.get("files"), msg.get("source"), msg.get("tables")
        options, preset = msg.get("options"), msg.get("preset")
        if not isinstance(files, dict) or sorted(files) != sorted(FLUX_FILE_ROLES):
            raise EngineFailure("files")
        if source != {}:
            raise EngineFailure("source")
        if not isinstance(tables, dict) or sorted(tables) != ["flux-vae"]:
            raise EngineFailure("files")
        if options != {} or not isinstance(preset, dict) or set(preset) != {"steps", "guidance"}:
            raise EngineFailure("config")
        steps, guidance = preset["steps"], preset["guidance"]
        if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 50 or not isinstance(guidance, (int, float)):
            raise EngineFailure("config")
        tokenizer = _tokenizer_files(files, FLUX_TOKENIZER_ROLES)
        _check_tables(tables)

        torch = runtime.prepare(msg, stage)
        self.torch = torch
        from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel
        from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
        from transformers import Qwen3Config, Qwen3ForCausalLM

        stage("flux-transformer")
        with empty_parameters(torch):
            tr = Flux2Transformer2DModel.from_config(dict(FLUX_TRANSFORMER))
        sd = convert_flux2_transformer_checkpoint_to_diffusers(model_bytes(files["flux-transformer"], "cuda"))
        tr.load_state_dict(sd, assign=True, strict=True)
        del sd
        tr = tr.to(torch.bfloat16).eval()

        stage("flux-vae")
        with empty_parameters(torch):
            vae = AutoencoderKLFlux2.from_config(dict(FLUX_VAE))
        table = _table(tables["flux-vae"]["path"])
        shapes = {k: tuple(v.shape) for k, v in vae.state_dict().items()}
        if sorted(shapes) != sorted(table):
            raise EngineFailure("files")
        raw = model_bytes(files["flux-vae"], "cuda")
        sd = {}
        for k, src in table.items():
            t = raw.pop(src, None)
            if t is None or t.numel() != _numel(shapes[k]):
                raise EngineFailure("files")
            # The mid-block attention projections are 1x1 convolutions in the
            # file and linear layers in Diffusers.
            sd[k] = t.reshape(shapes[k])
        if raw:
            raise EngineFailure("files")
        vae.load_state_dict(sd, assign=True, strict=True)
        del sd
        vae = vae.to(torch.bfloat16).eval()

        stage("qwen3")
        conf = Qwen3Config(**dict(QWEN3))
        conf._attn_implementation = "sdpa"
        with empty_parameters(torch):
            te = Qwen3ForCausalLM(conf)
        sd = model_bytes(files["qwen3"], "cuda")
        missing, unexpected = te.load_state_dict(sd, assign=True, strict=False)
        del sd
        if conf.tie_word_embeddings:
            te.tie_weights()
            missing = [m for m in missing if m != "lm_head.weight"]
        if missing or unexpected:
            raise EngineFailure("files")
        te = te.to(torch.bfloat16).eval()
        te.forward = runtime.phased("text_encode", te.forward)

        stage("pipeline")
        tok = load_tokenizer(tokenizer)
        sch = FlowMatchEulerDiscreteScheduler.from_config(dict(FLUX_SCHEDULER))
        # model_index.json at e7b7dc27: is_distilled, so no classifier-free
        # guidance.
        pipe = Flux2KleinPipeline(scheduler=sch, vae=vae, text_encoder=te, tokenizer=tok, transformer=tr, is_distilled=True)
        decode = vae.decode

        def decode_step(*a, **k):
            if self.on_decode is not None:
                self.on_decode(before=True)
            out = decode(*a, **k)
            if self.on_decode is not None:
                self.on_decode(before=False)
            return out

        vae.decode = decode_step
        self.pipe, self.steps, self.guidance = pipe, steps, float(guidance)
        torch.cuda.synchronize()
        log("loaded", set=self.name, gpu_reserved=torch.cuda.memory_reserved(), gpu_allocated=torch.cuda.memory_allocated())

    def unload(self):
        self.pipe = None

    def generate(self, job, report):
        torch = self.torch
        report = settled(torch, report)
        # Session.generate logged the job's start; one log line per step and
        # for the decode places a job that ends the process (MODELHOST-E1).
        # Never the prompt.
        log("plan", id=job.id, steps=self.steps)

        # Each end line follows the wait for the device, so a fault in the
        # step or the decode ends the log before its end line.
        def on_step_end(pipe, i, t, kwargs):
            torch.cuda.synchronize()
            log("step", id=job.id, step=i + 1, at="end")
            report("step", min(i + 1, self.steps), self.steps)
            return kwargs

        def on_decode(before):
            if before:
                job.check()
                log("decode", id=job.id, at="start", gpu_reserved=torch.cuda.memory_reserved())
                return
            torch.cuda.synchronize()
            log("decode", id=job.id, at="end")
            report("decode", 1, 1)

        self.on_decode = on_decode
        try:
            torch.cuda.reset_peak_memory_stats()
            g = torch.Generator(device="cpu").manual_seed(job.seed)
            job.check()
            with torch.no_grad():
                images = self.pipe(prompt=job.prompt, height=job.height, width=job.width, num_inference_steps=self.steps,
                                   guidance_scale=self.guidance, generator=g, output_type="pt",
                                   callback_on_step_end=on_step_end).images
            pixels = (images[0].permute(1, 2, 0).float() * 255).round().clamp(0, 255).to(torch.uint8).cpu()
            torch.cuda.synchronize()
            peak = int(torch.cuda.max_memory_reserved())
            log("image", width=job.width, height=job.height, gpu_peak_reserved=peak)
            if tuple(pixels.shape) != (job.height, job.width, 3):
                raise EngineFailure("error")
            return [pixels.contiguous().numpy().tobytes()], peak
        finally:
            self.on_decode = None


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def _decode_chunks(latent_frames):
    """How many decoder calls TurboVAED's temporal tiling makes: AMD's
    _temporal_tiled_decode loop for --t_block_size 16 --t_stride 12."""
    block, stride = DECODE_BLOCK // 4, DECODE_STRIDE // 4
    if latent_frames <= block:
        return 1
    n = 0
    for i in range(0, latent_frames, stride):
        n += 1
        if i + stride + 1 == latent_frames:
            break
    return n


def _accepted(cls, kwargs):
    import inspect

    ok = set(inspect.signature(cls.__init__).parameters) - {"self", "cls"}
    return {k: v for k, v in kwargs.items() if k in ok}


def _table(path):
    with open(path, encoding="utf-8") as f:
        table = json.load(f)["table"]
    if not isinstance(table, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in table.items()):
        raise EngineFailure("files")
    return table


def _sdpa_attention(torch):
    F = torch.nn.functional

    def attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, q_scale=None, causal=False,
                  window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16, version=None):
        if q_lens is not None or tuple(window_size) != (-1, -1):
            raise NotImplementedError("q_lens and window attention are not used by Hummingbird-XT")
        out_dtype = q.dtype
        if q_scale is not None:
            q = q * q_scale
        outs = []
        for i in range(q.size(0)):
            qi, ki, vi = q[i:i + 1], k[i:i + 1], v[i:i + 1]
            if k_lens is not None:
                n = int(k_lens[i])
                ki, vi = ki[:, :n], vi[:, :n]
            x = F.scaled_dot_product_attention(qi.transpose(1, 2).to(dtype), ki.transpose(1, 2).to(dtype),
                                               vi.transpose(1, 2).to(dtype), is_causal=causal, scale=softmax_scale)
            outs.append(x.transpose(1, 2))
        return torch.cat(outs).to(out_dtype)

    return attention


def serve(sets, runtime=None):
    rfile, wfile = _claim_stdio()
    Session(Wire(rfile, wfile), sets, runtime if runtime is not None else TorchRuntime()).run()


if __name__ == "__main__":
    serve({HummingbirdXT.name: HummingbirdXT(), Flux2Klein.name: Flux2Klein()})
