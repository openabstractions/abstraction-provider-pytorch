"""Checks the real engine script's unscale_weights against values built
here from the formats' definitions: per-tensor scaled float8_e4m3fn, and
NVFP4 with two E2M1 codes a byte (the first in the high nibble), one
float8_e4m3fn scale per 16 values in cuBLAS's tiled layout and a float32
tensor scale. Every value is a small dyadic number, so the expected BF16
weights are exact.

The PyTorch provider's tests run it as:
python unscale_reference.py <oa_torch_engine.py>. It needs torch; without it
it prints SKIP and exits 0.
"""
import importlib.util
import json
import random
import sys

try:
    import torch
except ImportError:
    print("SKIP no torch")
    sys.exit(0)

spec = importlib.util.spec_from_file_location("oa_torch_engine", sys.argv[1])
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
rng = random.Random(20261109)
failures = []


def check(name, got, want):
    if got.dtype != torch.bfloat16 or tuple(got.shape) != tuple(want.shape) or not torch.equal(got.float(), want.float()):
        failures.append(name)


def blocked(scales):
    """scales (rows, cols) in cuBLAS's block-scaling layout: 128x4 tiles in
    row-major tile order, the scale at (r, c) of a tile at offset
    (r % 32) * 16 + (r // 32) * 4 + c."""
    rows, cols = scales.shape
    row_tiles, col_tiles = -(-rows // 128), -(-cols // 4)
    flat = [0.0] * (row_tiles * col_tiles * 512)
    for r in range(rows):
        for c in range(cols):
            tile = (r // 128) * col_tiles + c // 4
            rr, cc = r % 128, c % 4
            flat[tile * 512 + (rr % 32) * 16 + (rr // 32) * 4 + cc] = float(scales[r, c])
    return torch.tensor(flat, dtype=torch.float32).reshape(row_tiles * 128, col_tiles * 4).to(torch.float8_e4m3fn)


def nvfp4(rows, blocks):
    """A packed NVFP4 weight, its tiled block scales, its tensor scale and
    the BF16 weight they encode."""
    cols = blocks * 16
    codes = [[rng.randrange(16) for _ in range(cols)] for _ in range(rows)]
    scales = torch.tensor([[2.0 ** rng.randrange(-3, 4) for _ in range(blocks)] for _ in range(rows)])
    tensor_scale = torch.tensor(0.5, dtype=torch.float32)
    packed = torch.tensor([[codes[r][2 * i] << 4 | codes[r][2 * i + 1] for i in range(cols // 2)] for r in range(rows)],
                          dtype=torch.uint8)
    want = torch.tensor([[E2M1[codes[r][c]] * float(scales[r, c // 16]) * 0.5 for c in range(cols)] for r in range(rows)])
    return packed, blocked(scales), tensor_scale, want.to(torch.bfloat16)


def record(fmt):
    return torch.tensor(list(json.dumps({"format": fmt}).encode()), dtype=torch.uint8)


# NVFP4 alone: 130 rows and 5 blocks pad the tiled scales to 256 x 8.
packed, scales, tensor_scale, want = nvfp4(130, 5)
check("nvfp4 padded tiles", engine.nvfp4_to_bf16(torch, packed, scales, tensor_scale), want)

# FP8 alone.
fp8 = torch.tensor([[1.0, -2.0, 0.5, 448.0], [0.0, -0.25, 3.0, -448.0]]).to(torch.float8_e4m3fn)
fp8_want = torch.tensor([[0.25, -0.5, 0.125, 112.0], [0.0, -0.0625, 0.75, -112.0]]).to(torch.bfloat16)
check("fp8", engine.fp8_to_bf16(torch, fp8, torch.tensor(0.25)), fp8_want)

# One state dict as the compact files hold it: a BFL FP8 layer with its
# input scale, a Comfy FP8 layer and NVFP4 layer with their records, a plain
# BF16 weight and a norm scale that stay as they are.
packed, scales, tensor_scale, nv_want = nvfp4(128, 4)
plain = torch.tensor([[1.5, -1.0]], dtype=torch.bfloat16)
norm = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
sd = {
    "a.weight": fp8.clone(), "a.weight_scale": torch.tensor(0.25), "a.input_scale": torch.tensor(1.0),
    "b.weight": fp8.clone(), "b.weight_scale": torch.tensor(0.25), "b.comfy_quant": record("float8_e4m3fn"),
    "c.weight": packed, "c.weight_scale": scales, "c.weight_scale_2": tensor_scale, "c.comfy_quant": record("nvfp4"),
    "d.weight": plain.clone(), "e.norm.scale": norm.clone(),
}
out = engine.unscale_weights(torch, sd)
if sorted(out) != ["a.weight", "b.weight", "c.weight", "d.weight", "e.norm.scale"]:
    failures.append("state dict keys %s" % sorted(out))
else:
    check("state dict fp8", out["a.weight"], fp8_want)
    check("state dict comfy fp8", out["b.weight"], fp8_want)
    check("state dict nvfp4", out["c.weight"], nv_want)
    check("state dict plain", out["d.weight"], plain)
    check("state dict norm", out["e.norm.scale"], norm)

# A plain BF16 file is returned unchanged.
plain_sd = {"x.weight": plain.clone()}
if engine.unscale_weights(torch, plain_sd) is not plain_sd or list(plain_sd) != ["x.weight"]:
    failures.append("plain file changed")

# Refusals: a scale without its weight, an FP8 weight without its scale,
# NVFP4 without its tensor scale and an unknown format.
for name, bad in (
    ("orphan scale", {"a.weight_scale": torch.tensor(1.0)}),
    ("fp8 without scale", {"a.weight": fp8.clone()}),
    ("nvfp4 without tensor scale", {"c.weight": packed, "c.weight_scale": scales, "c.comfy_quant": record("nvfp4")}),
    ("unknown format", {"c.weight": packed, "c.weight_scale": scales, "c.comfy_quant": record("int8_tensorwise")}),
):
    try:
        engine.unscale_weights(torch, bad)
        failures.append("accepted " + name)
    except engine.EngineFailure:
        pass

print(json.dumps({"failures": failures}))
sys.exit(1 if failures else 0)
