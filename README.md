# abstraction-provider-pytorch

**Status: development provider; not shipped.** The engine now starts in a
capability-less Windows AppContainer and its model files are lent by
read-only handles. The Setup component and CPU containment tests exist;
the reviewed GPU stack and installer still need their coordinated live run.

abstraction-provider-pytorch runs the OA PyTorch engine: an engine script OA
wrote, `go/torch/torch/oa_torch_engine.py`, on the reviewed PyTorch 2.11.0 /
ROCm 7.13.0 venv for gfx1151. It serves `abstraction.inference/video@1` for
AMD's Hummingbird-XT image to video, with the TurboVAED decoder as ASUS
MuseTree runs it, and `abstraction.inference/image@1` for Black Forest Labs'
FLUX.2 [klein] 4B, the start image of MuseTree's Idea Clip. OA supplies this
engine because no engine on the machine serves these models (VISION
2026-10-02, "OA may supply an engine process"); the model host keeps wrapping
engines the machine has. The engine moved out of the model host into this
provider on 2026-10-02 ([HISTORY.md](HISTORY.md)).

**What's new here:** Hummingbird-XT video and FLUX.2 [klein] images run on
an engine OA supplies, on Windows x64 with an AMD gfx1151 GPU, under a card
lease and reached through the runtime. The engine loads only pinned files in
place, its weights as safetensors. Its Python code makes no network call:
the Hugging Face hub client and its HTTP stack are stubs that refuse every
call, and the audit hook refuses the rest. The ROCm libraries that link
`ws2_32` still need the OS-level block before the engine ships.

Like every local engine, the provider is declared with `card:0`, loads each
set under its own lease through the resource kit's `holder`, composes the
resource kit's GPU fault Guard, and is reached through the runtime's router.
The engine process, its supervision, the video table and the placed model
files are the provider kit's (`abstraction-provider-kit`).

## Start

```
pytorchd engine check --python <venv>\Scripts\python.exe --source <HummingbirdXT>\infer
pytorchd models install --from C:\ProgramData\ASUS\AICreator ^
  --from <folder holding wan-diffusers\> --from <folder holding umt5-xxl\>
pytorchd models install --set flux2-klein-4b ^
  --from %USERPROFILE%\OA-models --from <folder holding flux2-klein-4b\tokenizer\>
pytorchd serve --endpoint pytorch --video-model hummingbird-xt-i2v --image-model flux2-klein-4b ^
  --python <venv>\Scripts\python.exe --source <HummingbirdXT>\infer
```

`pytorchd help` prints every option. The provider writes the script into
its state, proves the venv's interpreter, the script and AMD's eight source
modules before every lease, and speaks to the engine over its stdin and
stdout; the engine has no port and no socket. `models install` copies
nothing: it finds each pinned file in the `--from` folders, proves its size
and SHA-256 where it lies and records the path. It reads no other
application's store itself; a search through the runtime's storage service
is to come (TODO D5d).

The video set loads under a 31.0 GiB device lease and stays loaded between
videos, so a warm video pays no load. It delivers an H.264 MP4 the provider
encodes from the engine's frames with Windows' own encoder, the GPU's
hardware encoder when there is one. `--video-model-option
decode-dtype=bf16` turns on the four times faster BF16 decode, which stays
off until the owner has viewed its clips. The image set makes one 512x512
image in four steps under an 18.5 GiB lease, delivered as a PNG the provider
writes; it needs no AMD source. Both sets share one engine process, the
second holding the difference of their measured lease together.

## How the runtime uses it

`openabstractions models host` writes the declaration, beside the model
host's, after proving the engine with `pytorchd engine check`:

```
openabstractions models host --video-kind torch --video-model hummingbird-xt-i2v ^
  --image-model flux2-klein-4b ^
  --torch-python <venv>\Scripts\python.exe --torch-source <HummingbirdXT>\infer
```

The declaration is the native provider `pytorch`: `pytorchd serve` with
the runtime's endpoint, `card:0`, and the video and image profiles with
their models. It names a lease endpoint only when `--resource-endpoint`
does; otherwise the runtime that starts the provider names its own
(`ABSTRACTION_RESOURCE_ENDPOINT`). The card makes the runtime write the
program's `abstraction.resource/hold` rules on `card:0` and on `awake`, as
for every card provider; the provider holds the wake lease while a video or
an image runs (0.4.0's stated provisional exception, VISION 2026-10-02). A
model host declaration that still carries the PyTorch kind is split on the
next `models host` run: the chat models and an sd-server video stay with the
model host, and the PyTorch sets move to this declaration. Until then
`modelhostd serve` refuses that line and says so.

`models host --state DIR` names the model host's state folder and gives this
provider `DIR\pytorch` inside it, so an isolated runtime keeps its own
places, GPU quarantine and engine files apart from the person's default
folder. Without `--state` the declaration names none and `pytorchd` uses
`<user cache>\OpenAbstractions\providers\pytorch`. Record the sets in the
folder the declaration names: `pytorchd models install --state DIR\pytorch`
for an isolated runtime. Places that `modelhostd models install --kind
torch` recorded before the move are not migrated; run `pytorchd models
install --set SET --from FOLDER` once for each set.

## Setup

The `pytorchd setup` command builds this component from local inputs. The
installer's feature for fetching those inputs with consent remains to be
built. The component consists of:

- python.org's CPython 3.12.10 for Windows x64, the base interpreter the
  reviewed venv's `pyvenv.cfg` names;
- a venv of it, created without pip, built from the stack review's lean
  lock (`research/security-reviews/pytorch-rocm-windows-2.11.0-rocm7.13.0-lean.lock`):
  25 distributions with `--require-hashes` and `--no-deps`, among them torch
  2.11.0+rocm7.13.0, rocm-sdk-core and rocm-sdk-libraries-gfx1151 7.13.0,
  diffusers 0.37.1, transformers 4.54.1 and safetensors 0.6.2, installed
  from local wheels with no index. The hub client and its HTTP stack are
  absent; the engine answers their imports with stubs that refuse every
  call;
- AMD's Hummingbird-XT `infer/` source at commit `929e90a2`, fetched as
  GitHub's tarball with every file's Git blob checked;
- then `pytorchd engine check` and `pytorchd models install` for each set,
  and the `models host` declaration above.

`pytorchd setup install` takes the local pinned CPython folder, wheel folders,
state folder and optional AMD source folder, makes the component under its
own root, records each sandbox grant before applying it, and refuses a launch
until the install commits. `pytorchd setup remove` revokes those grants and
deletes the profile using the manifest; without that ownership record it
leaves the engine tree untouched. The installer feature and reviewed GPU live
run remain to be completed before shipping.

## Platforms

Windows x64 with an AMD gfx1151 GPU only. The reviewed stack is pinned for
`windows/amd64` alone, and the engine starts only inside its AppContainer,
which exists only on Windows; elsewhere every launch is refused
`engine:none`. Tests that run the stand-in engine skip off Windows; the
refusal, protocol and table tests run everywhere.

## Layout

- **`go/torch/`** — the provider: `engine.go` (the pinned stack, the engine
  files and their proof, the start), `proto.go` (the stdin and stdout
  protocol), `set.go` (the `TorchModelSets` rows), `shared.go` (the one
  process two sets share), `video.go` and `image.go` (the `video@1` and
  `image@1` work), `host.go` (the holder, the Guard and the video table),
  `serve.go` (the inference configuration), `place.go` (placing a set's files).
- **`go/torch/torch/`** — the engine script and the key tables it renames
  tensors by, built into `pytorchd`.
- **`go/torch/testdata/faketorch/`** — a stand-in engine speaking the
  protocol on the standard library alone, which the tests run.
- **`go/cmd/pytorchd/`** — the one binary this layer produces.

## Licence

Apache-2.0.
