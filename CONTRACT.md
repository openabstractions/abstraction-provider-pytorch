# provider-pytorch contract

The contracts this provider serves are
[`abstraction.inference/video@1` and `image@1`](https://github.com/openabstractions/abstraction-inference/blob/main/CONTRACT.md),
whose words bind every call; the rules below say what this provider does
around them. It runs the OA PyTorch engine, an engine script OA wrote,
because no engine on the machine serves these models (VISION 2026-10-02,
"OA may supply an engine process"). The engine process, its supervision, the
video operation table and the placed model files are the provider kit's
(`abstraction-provider-kit`,
KIT-P1..P6, KIT-V1..V6, KIT-M1..M4); the lease and the GPU fault handling are
the resource kit's (abstraction-resource RES-L1..RES-L4, RES-G1..RES-G7).
These rules moved here from the model host with the engine (its HISTORY.md,
2026-10-02); the model host keeps wrapping engines the machine has.

Binds: none (records only)

| Service | Models | Rules |
| --- | --- | --- |
| `video@1` | `hummingbird-xt-i2v` | T1..T10 |
| `image@1` | `flux2-klein-4b`, FLUX.2 [klein] 4B | T1..T5, T8, T10, T11 |

## Reading this page

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD",
"SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED", "MAY", and "OPTIONAL" in this
document are to be interpreted as described in BCP 14 [RFC2119] [RFC8174]
when, and only when, they appear in all capitals, as shown here.

The rule ids below, such as `PYTORCH-T1`, are what tests and refusals cite.
A retired id is never reused; [HISTORY.md](HISTORY.md) keeps it with the
release it left.

| Letter | Meaning |
| --- | --- |
| D | declared, leased, guarded and routed |
| T | the PyTorch engine |
| G | GPU faults |

## Declared, leased, guarded and routed

**[PYTORCH-D1] The provider is declared with its card.** `openabstractions
models host` with `--video-kind torch`, `--torch-python` or `--image-model` declares this
provider, `pytorch`, as its own native provider: `pytorchd serve` with its
endpoint, `card:0` among its resources and `video@1` or `image@1` among its
contracts. It names a lease endpoint only when the person named one; the
runtime that starts the provider names its own
(`ABSTRACTION_RESOURCE_ENDPOINT`, facade FAC-R8). A `--state` named for the
model host gives this provider that folder's `pytorch` folder. The runtime writes the program's
hold on `card:0` and on `awake` from that declaration, as for every card
provider (facade FAC-R3), and routes callers to it through the router; no
application reaches its endpoint.

**[PYTORCH-D2] A load takes a lease first.** Every set loads through the
resource kit's `holder` under its own lease of `card:0` for its measured
size (PYTORCH-T5), renewed while it is resident and released when it
unloads, yields or the provider stops. A lease the service refuses is the
set's refusal (`lease:*`); a lease that ends while the engine readies stops
the engine and loads once more.

## What it serves

With `--video-model` the provider serves `video@1` for one PyTorch model
set on the OA PyTorch engine, under the conditions of
`research/security-reviews/pytorch-rocm-windows-2.11.0-rocm7.13.0.md` and
`hummingbird-xt-i2v-files.md`. The placed files are KIT-M1 and M2; the queue,
the bounds, first frames, retention and cancel are the provider kit's video
table (KIT-V1..V6). With `--image-model` it serves `image@1` for one
PyTorch image set on the same engine (PYTORCH-T11), in the video set's
process when both run (PYTORCH-T10).

**[PYTORCH-T1] The engine is the provider's own script on the reviewed
stack.** The engine is `go/torch/torch/oa_torch_engine.py`, built into
`pytorchd`, written into `<state>/torch/engine` under a name bound to its
SHA-256, and proved and held there with the key tables it renames Wan-AI's
and Comfy-Org's tensors by, those of every set the engine serves. `--python` names the reviewed venv's interpreter: on
Windows x64 the venv launcher of CPython 3.12.10 by size and SHA-256, whose
`pyvenv.cfg` names Python 3.12.10, an absolute home and no system
site-packages, and is held with the bytes checked. The base interpreter
that launcher starts, in the folder `pyvenv.cfg` names as home, is pinned
too: `python.exe`, `python312.dll`, `python3.dll`, `vcruntime140.dll`,
`vcruntime140_1.dll`, and `DLLs\_socket.pyd` and `DLLs\select.pyd`, the
native modules the network audit hook's own import of `socket` loads before
the hook runs (`select` through `socket`'s import of `selectors`), by size
and SHA-256, held like the launcher. It runs
the held script with `-I -S -B`: no `PYTHON*` variable, no user site, no
site module and so no `.pth` file and no site-packages on the path, no
bytecode written. At start the script imports the standard library alone
and installs the network audit hook before any other import; at its first
load it appends the venv's site-packages, which the provider names, to its
own path and only then imports torch and safetensors, then a video set's
AMD reviewed modules (PYTORCH-T2) or an image set's Diffusers and
Transformers classes from the venv. Under `-S` the interpreter's `sys.prefix` stays the base
interpreter's. `torch.load` is replaced by a refusal and every model file
loads as safetensors (MODELBRIDGE-L20). The base interpreter's standard
library, the other native modules of its `DLLs` folder (`*.pyd`) and the
venv's site-packages, `__pycache__` included, are not proved at launch: an accepted risk of this release, proved at install
against the stack review's lock, with a RECORD-based launch check to come.
An interpreter, script, table or source file that is not the pin, and a
platform with no pinned stack, fail the video with reason `engine:none`
before any lease. `pytorchd engine check` runs the same
proof.

The network-capable packages the venv's libraries import, the Hugging Face
hub client and its HTTP stack (`huggingface_hub`, `requests`, `urllib3`,
`httpx`, `httpcore`, `h11`, `anyio`, `aiohttp`, `fsspec`, `certifi`,
`charset_normalizer`, `idna`), are answered by the script's own finder at
the front of `sys.meta_path`, installed before site-packages is on the path:
each imports as a stub module, the version, constants and decorators an
importer reads at import time are given, and every other name is a class
whose call raises `OANetworkDenied` naming it. `importlib.metadata` reports
the pinned version a dependency check reads, for these and for `filelock`,
whose module is absent. The venv holds none of them (the stack review's lean
lock). torch.hub's download calls raise `OANetworkDenied` as well, and the
audit hook refuses every `urllib.request` request and `http.client`
connection, a loopback one included: on Windows urllib takes the system
proxy from the registry, and a proxy on 127.0.0.1 would carry a request
out. A set's
tokenizer is built from its pinned `tokenizer.json` with the tokenizers
library and the special tokens, settings and chat template of its pinned
configuration files, with no class lookup and no hub path. Native code in
torch and ROCm is outside the finder's reach; the OS-level containment the
stack review requires holds it.

**[PYTORCH-T2] Source and model files are bound where they lie.** The
set's eight AMD modules (Hummingbird-XT `infer/` at `929e90a2`) are proved
in `--source` by size and SHA-256 and held. The engine imports
exactly those, by module name, from their held final paths and from source
bytes it hashes again, never from cached bytecode; any other `videox_fun`
module and the package `__init__` files are refused. Model files are one
`TorchModelSets` row each, placed by `pytorchd models install`
and proved and held as KIT-M1 states, a set folder under the models
root becoming a model folder as it does for an sd-server set; the engine is given each held
file's final path, and OA copies none. A set's tokenizer files (four for
`hummingbird-xt-i2v`, seven for `flux2-klein-4b`) lie in one folder, which
holds no other file name the tokenizer loader reads. `flux2-klein-4b` runs
no AMD source. Its weights are Comfy-Org's single files at `5f526678`
(`flux-2-klein-4b.safetensors` and `flux2-vae.safetensors` under Black
Forest Labs' tensor names, `qwen_3_4b.safetensors` under Hugging Face's),
its tokenizer BFL's at `e7b7dc27`, and its configurations constants of the
script. The engine builds the transformer through the reviewed Diffusers
0.37.1's FLUX.2 key conversion, the VAE through `torch/flux2-vae-keys.json`
(a 1x1 attention convolution reshaped to its linear layer), and the Qwen3
encoder by name with its tied output head, each loaded strictly. Every
tensor so built equals, byte for byte, a tensor of the files
research/live-runs/2026-10-02.md loaded (checked once on the CPU without
torch: the 169 transformer tensors by the conversion's own rules, the 250
VAE tensors after rounding the F32 file to BF16 as the engine's cast does,
and 396 of 398 encoder tensors by name; the other two, the last layer's
MLP, feed no hidden state the pipeline reads, which takes layers 9, 18 and
27).

**[PYTORCH-T3] The engine is reached through its own stdin and stdout.**
No port and no socket exists. Each message is one JSON header line of at
most 64 KiB; a header with `bytes` N is followed by exactly N raw bytes.
The provider sends `load`, `unload`, `generate`, `cancel` and `ping`; the
engine answers `hello` (protocol 2), `loading`, `ready`, `error`,
`unloaded`, `progress`, `frame`, `done`, `failed`, `cancelled` and `pong`,
each with its own fields only. `load`, `unload` and `generate` name a model
set, and `loading`, `ready`, `error` and `unloaded` name the set the
provider asked to load; a frame is exactly the request's width × height × 3
bytes, and a failure is one code of `load`, `files`, `source`, `config`,
`oom`, `device`, `input` or `error`. The first message outside this list
stops the engine. The engine points its
own fd 0 at the null device and fd 1 at stderr, so no library writes into
the protocol. stderr is the engine's log, `<state>/logs/torch.log`, at most
8 MiB with one earlier file, never forwarded to a caller. inference INF-EL2's
socket check runs at readiness and every 30 seconds: an engine tree holding
any TCP or bound UDP socket is stopped with reason `engine:network-socket`.

**[PYTORCH-T4] The engine environment is built from nothing.** The
provider kit's clean environment (inference INF-EL1) with the research
launcher's computed names (`USERNAME`, `PROCESSOR_IDENTIFIER`, `OS`,
`ComSpec`, `PATHEXT` beside L6's), `PATH` of the venv's scripts, System32
and the Windows folder, the offline hub variables, every cache under
`<state>/torch/cache`, `TEMP` and `TMP` in `<state>/torch/temp`, the
working folder `<state>/torch/work`, `ROCM_SDK_TARGET_FAMILY=gfx1151`, and
the resource kit's `pytorch-rocm` dispatch row's variables (RES-G5:
`MIOPEN_FIND_MODE=FAST`, `MIOPEN_DEBUG_CONV_DIRECT=0`, the naive solvers
off with `MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD`, `_BWD` and `_WRW` `=0`,
the Winograd Fury solvers off, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`). `PYTHON*`,
`ROCM_SDK_PRELOAD_LIBRARIES`, `HIP_VISIBLE_DEVICES`, `LD_*`, `HF_TOKEN`, every
proxy variable and every name outside the list are refused by name. The
row's engine-side fixes reach the engine in its load, which applies each:
MIOpen convolution off (`torch.backends.cudnn.enabled=False`) and
Hummingbird-XT's `seq_len` from the clip, never 27,280. A row fix the
engine does not apply refuses every launch with reason `engine:none`.
Before the lease and again at every launch, the provider kit's GPU library
path gate (inference CONTRACT.md INF-EL3) reads the venv's hipBLASLt and
rocBLAS `library\gfx1151` folders; a venv too deep for their kernel paths
with Windows long paths off runs through verified 8.3 names in
`HIPBLASLT_TENSILE_LIBPATH` and `ROCBLAS_TENSILE_LIBPATH`, or fails the video
with reason `engine:gpu-path` before any lease.

**[PYTORCH-T5] The engine is resident under its measured lease.** The
first video loads the engine as one serialized step of its holder (PYTORCH-D2):
`Acquire` for the set's measured device memory, 31.0 GiB for
`hummingbird-xt-i2v` (the 512×512×81 peak of 31.51 GB reserved with AMD's
F32 decode in the diagnostic rerun of 2026-10-02,
`feedback/gpu-runs-2026-10-02.md` item 22, with about 5 percent above it)
and 18.5 GiB for
`flux2-klein-4b` (the 512×512 peak of 18.83 GB reserved in the live
walkthrough of 2026-10-02, `research/live-runs/2026-10-02-walkthrough.md`
D9, with about 5 percent above it), then
the engine. It is ready when it sends `ready` within `--ready`. It
stays loaded between videos and leaves on the idle time, a yield, an unload
or its exit, each releasing the lease. The holder admits it beside chat
engines while their leases fit, and refuses it with `lease:insufficient`
before it starts when they do not. The job's per-process memory limit is
the set's measured peak commit with a margin, 43 GiB for
`hummingbird-xt-i2v` and 25 GiB for `flux2-klein-4b`, unless
`--memory-limit` names one. A job whose peak reserved memory passes
what the process's resident sets lease is reported through the provider's
error channel; the residency stays charged (RES-L2).

**[PYTORCH-T6] The provider owns the job, its progress and its cancel.**
A prompt holding a control character, U+2028 or U+2029 is refused at
`Start` with `unsupported_feature` and reason `video:engine-bound`, by the
same bound every sd-server request body applies (`sdcpp.CheckPrompt`), so
no prompt forges a line of the engine's log. The engine receives the set's preset (`hummingbird-xt-i2v`: AMD's
timesteps 1000, 750 and 500 at shift 10, guidance 1, boundary 0.875), the
prompt, the seed, the size, the frames and the provider's first frame as
raw RGB of the request's size. Progress arrives per denoising step and per
decode chunk once the device has finished it (the engine waits for the
device before each report): `Observe` carries `step` of 100, the steps filling the set's
share for its decode precision (48 with F32, 76 with BF16) and the decode
the rest, at most 99 until the result is kept. A `Cancel` of a running
video reaches the engine, which stops before its next step or decode chunk
and stays resident; an engine that has not stopped within 60 seconds is
stopped. A video fails `engine:refused` for an input the engine refuses,
`engine:failed` for `oom` or `error`, `engine:lost` when the engine exits or
breaks the protocol, `engine:result` for frames that differ from the
request, and `engine:timeout` past `--video-time-limit`, whose engine is
stopped. No reason carries engine text.

**[PYTORCH-T7] The container is the provider's own.** After checking
the frame count and every frame's size, the provider encodes the engine's
raw frames as one H.264 MP4 (`video/mp4`) at the request's rate through the
provider kit's `providerhost/videoenc` (inference INF-VE1): the hardware
H.264 encoder when the machine has one, and Microsoft's software encoder
when there is none or the hardware encoder fails. No container the engine
writes reaches a caller. The MP4 is kept as one local delivery (INF-G3),
and the completed operation's `Observe` reason names the encoder that ran,
`encoder:hardware` or `encoder:software`. A video no system encoder can
write (options the encoder refuses, or a platform without one) fails
`encoder:unavailable`; any other encode failure, a disk or file error
included, fails `encoder:failed` and is reported in its own words through
the provider's error channel. No caller asks for another container,
and the provider writes none.

**[PYTORCH-T8] GPU faults reach the Guard.** The engine reports a lost
device as one line on its log, `oa-torch-engine: device-lost: ` and a HIP
error name (`providerhost/devicelost` `PyTorchEngine`), sends `failed` with
code `device` and exits; the line counts once the engine exits or fails its
ping twice (RES-G1). The wake lease and the standby refusal are the
holder's (PYTORCH-G1); the engine holds neither. The quarantine
fingerprint is the model with `set=torch:<set>` and its options, the
engine script's digest with the pinned stack, the size, frames, rate and
whether a first frame conditions it. A request whose clip token count
reaches the dispatch row's known-bad value is refused at `Start` with
`unavailable` and reason `gpu:known-bad`.

**[PYTORCH-T9] Model-set options.** `--video-model-option NAME=VALUE`
chooses among the row's options; `hummingbird-xt-i2v` takes
`decode-dtype=f32` (the default, AMD's) or `bf16` (a decode about four times
faster, research run B2), which stays opt-in until the owner has viewed its
clips. An option or value the row does not offer exits 2.

**[PYTORCH-T10] Two sets share one engine process.** With
`--video-model` and `--image-model`, the video set and the image set
are two holder components of one engine process, as MuseTree's engine
holds FLUX.2 [klein] and Hummingbird-XT. Either set loads first and starts
the process under its own lease, to which the process is bound; the other
joins the running process with `load` and holds a second lease, bound to
no process (one child binds to one lease, RES-L2), of the two sets'
measured lease together less the first set's own: for `flux2-klein-4b`
beside `hummingbird-xt-i2v`, 40.9 GiB with the BF16 decode and 44.4 GiB
with the F32 decode (research/live-runs/2026-10-02.md phase 1; run E of
2026-10-01 reserved 40.5 GiB), so the image set joining the video set
holds 9.9 or 13.4 GiB, and the video set joining the image set 23.4 or
26.9 GiB. Two sets never measured together each hold
their own lease. `resource/leases@1` grants an amount once and has no
resize (RES-L1, RES-L2), so the process's reservation grows by the joined
set's lease and shrinks by giving it back; a swap to one larger lease would
ask the service for the whole amount again while the first lease is still
held. Every engine operation holds the process's one operation slot: a
load (the process's start or a set joining it), an unload and a job, so one
runs at a time and an unload waits for a running job rather than queueing
behind it in the engine. While an operation holds the slot every set's
component is in use, and the first set's component is in use while either
set has work queued or running, so neither the idle time nor a yield takes
a set from under a job. The lock order is a set's holder component lock,
then the slot: a job holds its set's use and has the set loaded before it
holds the slot, and a load or an unload takes the slot inside the component
lock the holder holds through it. A driver reset takes no slot: the
process is stopped at once, before the holder stops each component
(RES-G2). Unloading the joined set, for the idle time, a yield or an
unload, sends `unload` once it holds the slot: the engine drops the set,
empties its cache and answers `unloaded`, the set's files are let go, and
its lease is released, which leaves the first set's own. A join whose wait
ends before the set is ready, a cancelled job's included, still holds its
own operation: its Stop sends `unload` within it, and the lease goes back
only after the engine's `unloaded`, its refusal of the load, or the
process's exit, so no set stays loaded without a lease. An engine that
does not answer within 20 seconds is stopped. Unloading the first set
stops the process; the holder releases the joined set's lease at its next
check of that set, at the latest one renewal interval later. A set that
fails to join is refused with the engine's code; the first set serves on,
and the refused set's lease goes back. With both
sets configured the process's memory limit is the two sets' commit
together with its margin, 55 GiB (42.6 and 48.3 GB measured), unless
`--memory-limit` names one.

**[PYTORCH-T11] Images are served on the PyTorch engine.** `--image-model
SET` serves `image@1` for one PyTorch image set, `flux2-klein-4b`, through
the inference provider's composed image engine (inference
`Config.ImageEngine` on the router host `pytorch-image`); the
provider keeps admission, bounds and the caller's publication, and its
Admit lets the image model through without a load. One request makes one
image from its prompt alone at a size the set measured (512×512, size
`auto`); an edit, a count above 1, another size and any extension are
refused `unsupported_feature` before any rights check, and a prompt
holding a control character, U+2028 or U+2029 with reason
`video:engine-bound`, by the same bound as PYTORCH-T6. The image waits
while the machine is in Modern Standby or the driver comes back from a
reset, and is refused `unavailable` with `gpu:quarantined` when its
fingerprint is quarantined or `gpu:known-bad` at the dispatch row's
known-bad token count (one token per 16×16 pixels and 512 text tokens);
the wake lease is held while it runs. The engine receives the preset (four
steps, guidance 1), the prompt, a seed the provider draws and the size;
progress arrives per step and for the decode once the device has finished
each, the steps filling 85 of 100 and the decode the rest.
A cancel reaches the engine, which stops before its next step and stays
resident; one that has not stopped within 30 seconds is stopped. The
engine's one frame of raw RGB becomes the provider's own PNG through the
provider kit (inference INF-IO1), published through the provider's content
writer into its content store with a 20 MiB bound. An image fails
`unavailable` with `engine:refused`, `engine:failed`, `engine:lost`,
`engine:result`, `engine:timeout` past `--image-time-limit` (default 10
minutes) or `gpu:reset`, as PYTORCH-T6 and T8 read for a video; no reason
carries engine text. The quarantine fingerprint is the model with
`set=torch:<set>`, the engine script's digest with the pinned stack, the
size and the step count.

**[PYTORCH-T12] The supplied engine runs in a bounded Windows sandbox.**
Setup builds the pinned interpreter and lean wheel lock from local inputs,
records its capability-less AppContainer profile and every file grant before
applying it, and makes the sandbox ready only when installation completes
(KIT-S1..S5). Every Windows launch checks that ready profile and its current
grants before starting inside it; an absent or busy sandbox refuses with
`engine:sandbox`. The process becomes the holder's owned component before
hello and first load; a failure there is a readiness failure whose lease
stays held until Stop confirms exit and cleanup. The model files have no
path grant. After a bounded hello,
the actual Python receiver receives read-only handles only when its live
process handle proves membership in the owned job and the exact container
SID (FAC-P4, KIT-S6). The engine and every descendant have no network
capabilities. Holder Stop completes only after post-exit model/engine holds,
the sandbox launch lock and the log have finished cleanup. A completed close
error is reported through the provider's error callback and still lets the
lease go; if cleanup exceeds its budget, Stop reports failure and the lease
remains held. CPU tests cover
local loopback denial, an admitted OA pipe exchange and a denied competing
server instance, nested grant removal and immediate Setup after Host.Close.

## GPU faults

**[PYTORCH-G1] The provider applies the resource kit's GPU fault rules.**
`pytorchd serve` MUST compose the resource kit's provider Guard
(`gpufault.ForProvider`) over its holder: on Windows it reads the platform's
reset and standby records, and on every platform the engine's own
device-lost line, counted only once the engine exits or fails its readiness
twice (RES-G1, KIT-P4). A reset stops the engine's process at once, then
every set through the holder, and releases their leases (RES-G2). A video or
image whose engine was lost in a reset fails `gpu:reset` and its fingerprint
is quarantined (PYTORCH-T8, RES-G3). A load waits while the machine is in
Modern Standby or the driver is coming back (RES-G4, RES-G5). While a video
or an image runs the provider holds the wake lease, as every card provider
does (0.4.0's stated provisional exception, VISION 2026-10-02). The
quarantine lives in `gpu-quarantine.json` under `--state`; `pytorchd gpu
list` prints it and `pytorchd gpu clear` releases a paused request. Each
fault is one OA logging record from `pytorchd` with its Panel line (RES-G6).

## Engine exits

Every engine exit is recorded as the provider kit records it (KIT-P3), from
`pytorchd`, prefixed `pytorch:` in the engine's log. The engine script enables `faulthandler` on its log before any extension module
is imported, so a native fault prints every thread's Python stack. It logs
one line per load stage, one when a set unloads, one when a job starts, and
one per phase of every set: for video the text encode and the first
frame's encode (start and end), each sampling step, the decode (start and
end) and each decode chunk; for an image the text encode (start and end),
each step's end and the decode (start and end). These lines never carry
the prompt. The exit
record's code and run time are the authority on how and when an engine
ended. On Windows `faulthandler` may also print a trace for a native
exception a library raised and handled itself, with the engine running on;
such a trace without an exit record is no crash.

## Outcomes

| Call | Outcome | Reason | Meaning |
| --- | --- | --- | --- |
| `image@1` | the OA PyTorch engine (`--image-model`) | `flux2-klein-4b`, FLUX.2 [klein] 4B | T1..T5, T8, T10, T11 |
| `video@1` `Start` | `unavailable` | `gpu:known-bad` | On the PyTorch engine, the clip's token count reaches the value the dispatch row records as hanging the GPU (PYTORCH-T8). |
| `video@1` `Observe` | `failed` | `engine:none`, `engine:gpu-path`, `engine:unavailable`, `engine:network-socket`, `engine:sandbox`, `lease:*` | On the PyTorch engine: the interpreter, script, source or a placed file is not the pin, the venv is too deep for its GPU kernel paths, the sandbox is absent or changed, the engine failed its load or held a socket, or the lease was refused (PYTORCH-T1..T5, T12). |
| `image@1` | `unsupported_feature` | `model:<name>`, `mode:edit`, `count`, `size`, `extension:<key>`, `prompt`, `video:engine-bound` | The PyTorch image set makes one image from a prompt at a measured size; the prompt is empty or beyond the sd path's bound (PYTORCH-T11). |
| `image@1` | `unavailable` | `engine:none`, `engine:gpu-path`, `engine:unavailable`, `engine:network-socket`, `engine:sandbox`, `lease:*` | As for a PyTorch video, or the set failed to join the running engine (PYTORCH-T10..T12). |
| `image@1` | `unavailable` | `engine:refused`, `engine:failed`, `engine:result`, `engine:timeout`, `engine:lost` | The engine refused the request, failed, sent a result that differs from it, ran past `--image-time-limit`, or stopped answering (PYTORCH-T11). |
| `image@1` | `unavailable` | `gpu:quarantined`, `gpu:known-bad`, `gpu:reset` | The image is paused after a reset it caused, its shape is recorded as hanging the GPU, or the engine was lost in a reset (PYTORCH-T11, G1). |

## Development boundary

The provider downloads no interpreter or model. `pytorchd setup install`
builds a component from the local pinned CPython folder and the lean wheel
lock, then records and applies the AppContainer's engine and state grants.
Every Windows launch requires the ready sandbox. The contained interpreter
and its descendants have no network capability; the model files remain
outside its path grants and reach the actual Python receiver as read-only
handles, after the receiver's hello and process membership check. The
installer feature and coordinated live GPU run remain before shipping.
