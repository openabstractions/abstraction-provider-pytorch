"""Model files loaded onto the GPU one after another through the engine
script's own loader (model_bytes), timed, for TestMeasureModelLoad.

    python -I -B child.py ENGINE_SCRIPT RECEIVER SIZE:SHA256 [SIZE:SHA256 ...]

For each file it prints "need I" and reads the URI or path the parent
lends on one stdin line, as the engine asks the provider for a file when
its load reaches it. Prints one JSON line at the end.
"""
import ctypes
import hashlib
import importlib.util
import json
import sys
import time
from ctypes import wintypes


class Counters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]


class Perf(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                ("HandleCount", wintypes.DWORD), ("ProcessCount", wintypes.DWORD), ("ThreadCount", wintypes.DWORD)]


def counters():
    c = Counters()
    c.cb = ctypes.sizeof(c)
    k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError("GetProcessMemoryInfo")
    return c


def system_cache():
    p = Perf()
    p.cb = ctypes.sizeof(p)
    ctypes.WinDLL("psapi").GetPerformanceInfo(ctypes.byref(p), p.cb)
    return p.SystemCache * p.PageSize


def main():
    engine_path, receiver_path = sys.argv[1:3]
    files = [tuple(a.split(":", 1)) for a in sys.argv[3:]]
    spec = importlib.util.spec_from_file_location("oa_torch_engine_measure", engine_path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    with open(receiver_path, "rb") as f:
        receiver_sha = hashlib.sha256(f.read()).hexdigest()
    if hasattr(engine, "load_receiver"):
        engine.load_receiver({"path": receiver_path, "sha256": receiver_sha})
    import torch
    torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    cache0, before = system_cache(), counters()
    held, seconds, start = [], [], time.perf_counter()
    for i, (size, sha) in enumerate(files):
        print("need", i, flush=True)
        arg = sys.stdin.readline().strip()
        t0 = time.perf_counter()
        entry = {"size": int(size), "sha256": sha}
        entry["uri" if arg.startswith("fd:") else "path"] = arg
        held.append(engine.model_bytes(entry, "cuda"))
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - t0)
    total = time.perf_counter() - start
    after = counters()
    size = sum(int(s) for s, _ in files)
    print(json.dumps({"seconds": seconds, "engine_total": total, "bytes": size,
                      "device_bytes": sum(t.numel() * t.element_size() for sd in held for t in sd.values()),
                      "peak_working_set": after.PeakWorkingSetSize, "private_before": before.PrivateUsage,
                      "private_after": after.PrivateUsage, "file_cache_growth": system_cache() - cache0}), flush=True)


main()
