# History

Design records for [`CONTRACT.md`](CONTRACT.md).

## 2026-10-02: the engine leaves the model host

The OA PyTorch engine was the model host's PyTorch kind, `--video-kind torch`
and `--image-model` of `modelhostd`, built there between 2026-10-01 and
2026-10-02 (the model host's HISTORY.md records its reviews and live runs).
The architecture review of 2026-10-02 found it against VISION 2026-09-22
(4), that the model host wraps an engine the machine already has (D5). The
owner decided that OA may supply an engine process when no engine on the
machine serves a model, as its own provider module that registers like any
other engine (VISION 2026-10-02).

The code moved with its history (`git mv`): `torchengine.go`, `torchproto.go`,
`torchset.go`, `torchshared.go`, `torchvideo.go`, `torchimage.go`, the engine
script with its key tables, the stand-in engine and their tests. What the
PyTorch kind shared with the model host's other kinds moved first into the
provider kit (`abstraction-provider-kit`): the engine process launch, the
video@1 table, model file placement and the refusals.

The rules moved and were renumbered: MODELHOST-T1..T11 are PYTORCH-T1..T11
word for word except for the command and option names (`pytorchd`,
`--python`, `--source`, `--memory-limit`, `--ready`); the PyTorch part of
MODELHOST-G1 is PYTORCH-G1; MODELHOST-E1 is the provider kit's KIT-P3.
PYTORCH-D1 and D2 are new: the provider's own declaration with `card:0` and
its loads under the holder. The engine script is byte for byte the one the
model host ran, so its comments still cite the MODELHOST-T ids; its fingerprint
digest is unchanged. Engine diagnostics now read `pytorch:` and the image
router host is `pytorch-image`.

## 2026-10-02: the walkthrough's lease and progress fixes follow the move

Two fixes of the live walkthrough of 2026-10-02 landed on the model host's
PyTorch kind while it moved here, and moved with the merge
(`research/live-runs/2026-10-02-walkthrough.md`; modelhost HISTORY.md
records the measurements). D9: `flux2-klein-4b` leases 18.5 GiB, its
measured 18.83 GB peak with about 5 percent above it, in place of 17.5 GiB.
D10: the engine waits for the device before each step and decode report, so
progress fills as the work finishes. D13: `pytorchd models install` names the
`openabstractions models host` flags that declare the set it recorded.

| id | before | now |
| --- | --- | --- |
| PYTORCH-T5 | `flux2-klein-4b` leases 17.5 GiB | 18.5 GiB |
| PYTORCH-T6 | progress per denoising step and decode chunk | the same, reported once the device has finished each |
| PYTORCH-T11 | progress per step and for the decode, steps 85 of 100 | the same, reported once the device has finished each; the decode fills the rest |

Places that `modelhostd models install --kind torch` recorded in the model
host's state folder are not migrated: run `pytorchd models install --set
SET --from DIR` once for each set, which records them in the PyTorch
provider's own state folder.

## 2026-10-02: lean dependencies

The owner asked why the engine requires the Hugging Face libraries and
directed the third-party set leaned as far as it goes. An import trace of
every set's load and generate path on the CPU found 20 of the venv's 45
distributions unused once the hub client is answered by stubs: the hub
client and its whole HTTP stack, accelerate, scipy, networkx, psutil,
filelock, importlib_metadata, zipp, setuptools and pip. The engine now
answers the network-capable packages with stub modules that raise
`OANetworkDenied` on any call, builds tokenizers from their pinned files
with the tokenizers library, and refuses torch.hub's downloads by name. The
stack review's addendum "Lean dependencies" records the inventory, the
trace, the lean lock and what network-capable code remains.

| id | before | now |
| --- | --- | --- |
| PYTORCH-T1 | the venv's 44 locked distributions; tokenizers through `AutoTokenizer` | 25; the network-capable packages are stubs raising `OANetworkDenied`; tokenizers from `tokenizer.json` through the tokenizers library |
| PYTORCH-T1 | the audit hook refuses non-loopback sockets | it also refuses every `urllib.request` request and `http.client` connection, loopback included, since a system proxy on 127.0.0.1 carried `urlopen` out (review of 2026-10-02) |

