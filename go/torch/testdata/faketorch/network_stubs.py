"""Imports the real engine script as a module (its __main__ guard keeps it
from serving) in an interpreter that has none of the network-capable
packages installed, and reports, one JSON object, what the deny finder
gives: imports that succeed, values importers read, calls that raise
OANetworkDenied, versions a dependency check reads, and names recorded.

The PyTorch provider's tests run it as: python network_stubs.py <oa_torch_engine.py>.
"""
import importlib
import importlib.metadata
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("oa_torch_engine", sys.argv[1])
eng = importlib.util.module_from_spec(spec)
sys.modules["oa_torch_engine"] = eng
spec.loader.exec_module(eng)

out = {"imports": {}, "calls": {}, "versions": {}, "values": {}}
for name in ("huggingface_hub", "huggingface_hub.utils", "huggingface_hub.constants", "requests", "requests.exceptions",
             "urllib3", "httpx", "fsspec.implementations.http", "aiohttp", "certifi", "idna"):
    m = importlib.import_module(name)
    out["imports"][name] = getattr(m, "__file__", "")
hub = importlib.import_module("huggingface_hub")
utils = importlib.import_module("huggingface_hub.utils")
constants = importlib.import_module("huggingface_hub.constants")
out["values"] = {"version": hub.__version__, "offline": constants.HF_HUB_OFFLINE, "endpoint": constants.ENDPOINT,
                 "progress_bars_disabled": utils.are_progress_bars_disabled()}


@utils.validate_hf_hub_args
def decorated(x):
    return x + 1


out["values"]["decorator_passes_through"] = decorated(1) == 2


class Sub(utils.HfHubHTTPError):
    pass


out["values"]["subclass_is_exception"] = issubclass(Sub, Exception)
for label, call in (("hf_hub_download", lambda: hub.hf_hub_download("org/model", "config.json")),
                    ("snapshot_download", lambda: hub.snapshot_download("org/model")),
                    ("requests.get", lambda: importlib.import_module("requests").get("http://example.invalid/")),
                    ("requests.Session", lambda: importlib.import_module("requests").Session()),
                    ("httpx.Client", lambda: importlib.import_module("httpx").Client()),
                    ("subclass", lambda: Sub("x"))):
    try:
        call()
        out["calls"][label] = "returned"
    except eng.OANetworkDenied as e:
        out["calls"][label] = str(e)
for dist in ("huggingface-hub", "huggingface_hub", "requests", "filelock", "fsspec"):
    try:
        out["versions"][dist] = importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        out["versions"][dist] = None
try:
    importlib.import_module("filelock")
    out["filelock_import"] = "imported"
except ImportError:
    out["filelock_import"] = "missing"
# The standard library's HTTP clients are refused outright, a loopback
# target and a loopback proxy included (a system proxy on 127.0.0.1).
import http.client  # noqa: E402
import os  # noqa: E402
import urllib.request  # noqa: E402

out["stdlib"] = {}
os.environ["http_proxy"] = "http://127.0.0.1:9"
for label, call in (("urlopen through a loopback proxy", lambda: urllib.request.urlopen("http://example.invalid/", timeout=2)),
                    ("urlopen loopback", lambda: urllib.request.urlopen("http://127.0.0.1:9/", timeout=2)),
                    ("http.client loopback", lambda: http.client.HTTPConnection("127.0.0.1", 9, timeout=2).connect())):
    try:
        call()
        out["stdlib"][label] = "connected"
    except PermissionError as e:
        out["stdlib"][label] = str(e)
    except Exception as e:  # noqa: BLE001
        out["stdlib"][label] = "not refused: " + type(e).__name__
out["uses"] = sorted(eng.STUB_USES)
print(json.dumps(out))
