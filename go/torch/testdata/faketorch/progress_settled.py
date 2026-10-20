"""Runs the real engine script's FLUX.2 [klein] and Hummingbird-XT
generate with a stand-in torch and pipeline, and prints the order in which
the device was waited for and progress was reported, one JSON list.

The PyTorch provider's tests run it as: python progress_settled.py <oa_torch_engine.py>.
"""
import contextlib
import importlib.util
import json
import sys
import threading

spec = importlib.util.spec_from_file_location("oa_torch_engine", sys.argv[1])
real = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real)

EVENTS = []

# The engine's end lines of a step, a decode or a decode chunk are recorded
# beside the waits and reports, so their order can be read.
_log = real.log


def _recording_log(event, **kw):
    if kw.get("at") == "end" and event in ("step", "decode", "decode_chunk"):
        EVENTS.append(["end", event])
    _log(event, **kw)


real.log = _recording_log


class Tensor:
    """Every operation the generate code applies returns the tensor itself."""

    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, i):
        return self

    def __mul__(self, other):
        return self

    def tobytes(self):
        return b"\0"

    def unsqueeze(self, *a):
        return self


for name in ("permute", "float", "round", "clamp", "to", "cpu", "contiguous", "numpy", "view", "div", "sub"):
    setattr(Tensor, name, lambda self, *a, **k: self)


class Cuda:
    def synchronize(self):
        EVENTS.append(["sync"])

    def reset_peak_memory_stats(self):
        pass

    def max_memory_reserved(self):
        return 1

    def memory_reserved(self):
        return 1


class Generator:
    def __init__(self, device=None):
        pass

    def manual_seed(self, seed):
        return self


class Torch:
    cuda = Cuda()
    uint8 = "uint8"
    bfloat16 = "bfloat16"
    Generator = Generator

    @staticmethod
    def no_grad():
        return contextlib.nullcontext()

    @staticmethod
    def frombuffer(data, dtype=None):
        return Tensor(None)


def report(phase, done, total):
    EVENTS.append(["report", phase, done, total])


class Job:
    id = "job"
    prompt = "a lighthouse"
    seed = 1
    width = 64
    height = 64
    frame = b"\0" * (64 * 64 * 3)

    def __init__(self, frames):
        self.frames = frames
        self.cancel = threading.Event()

    def check(self):
        pass


# FLUX.2 [klein]: the pipeline calls the step callback per step, then the
# VAE's decode, which load wraps to call on_decode around it.
image = real.Flux2Klein()
image.torch, image.steps = Torch, 4


class ImagePipe:
    def __call__(self, **kw):
        for i in range(image.steps):
            kw["callback_on_step_end"](self, i, 0, {})
        image.on_decode(before=True)
        image.on_decode(before=False)
        return type("Out", (), {"images": Tensor((64, 64, 3))})()


image.pipe = ImagePipe()
image.generate(Job(1), report)

# Hummingbird-XT: the transformer wrapper calls on_step around each forward
# and the decoder wrapper on_decode around each chunk.
video = real.HummingbirdXT()
video.torch = Torch
video.preset = {"timesteps": [1000, 750, 500], "guidance": 1, "boundary": 0.875, "shift": 10}
frames = 17


class VideoPipe:
    def __call__(self, *a, **kw):
        for _ in video.preset["timesteps"]:
            video.on_step(before=True)
            video.on_step(before=False)
        return type("Out", (), {"videos": Tensor(None)})()


class Turbo:
    def ours_decode(self, pipe, tv, lat):
        for _ in range(real._decode_chunks((frames - 1) // 4 + 1)):
            video.on_decode(before=True)
            video.on_decode(before=False)
        return Tensor((frames, 64, 64, 3))


video.pipe, video.turbo = VideoPipe(), Turbo()
video.generate(Job(frames), report)
print(json.dumps(EVENTS))
