// Command pytorchd is the OA PyTorch engine provider: it runs the provider's
// own Python engine script on the reviewed PyTorch for ROCm venv and serves
// abstraction.inference/video@1 and image@1 for the reviewed PyTorch model
// sets, each set under its own lease of the card. OA supplies this engine
// because no engine on the machine serves these models (VISION 2026-10-02).
//
// It bundles no interpreter and downloads nothing.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/openabstractions/abstraction-identity/listen"
	inference "github.com/openabstractions/abstraction-inference/go"
	"github.com/openabstractions/abstraction-inference/go/providerhost"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
	logging "github.com/openabstractions/abstraction-logging/go"
	"github.com/openabstractions/abstraction-provider-kit/go/engineproc"
	"github.com/openabstractions/abstraction-provider-kit/go/modelplace"
	"github.com/openabstractions/abstraction-provider-kit/go/quarantine"
	"github.com/openabstractions/abstraction-provider-kit/go/runtimeprog"
	resource "github.com/openabstractions/abstraction-resource/go/client"
	"github.com/openabstractions/abstraction-resource/go/gpufault"
	"github.com/openabstractions/abstraction-resource/go/holder"

	"github.com/openabstractions/abstraction-provider-pytorch/go/torch"
)

const usage = `pytorchd runs the OA PyTorch engine: the provider's own Python script on
the reviewed PyTorch 2.11.0 / ROCm 7.13.0 venv
(research/security-reviews/pytorch-rocm-windows-2.11.0-rocm7.13.0.md), spoken
to over its stdin and stdout with no port or socket. It serves
abstraction.inference/video@1 and image@1 for the reviewed PyTorch model sets,
each set under its own lease of card:0 of abstraction.resource/leases@1, and
stops the engine when the card is asked for, after its idle time, or on a
graphics driver reset.

It bundles no interpreter and downloads nothing. pytorchd setup install
builds the engine from local files once: a copy of python.org's CPython
3.12.10, a venv of it holding the stack review's lean lock, and the engine's
sandbox, a Windows AppContainer with no network capability and the file
grants the engine needs (CONTRACT.md PYTORCH-T12). Every launch checks the
sandbox and starts the engine inside it, and refuses with engine:sandbox
when it is not in place; the interpreter, the engine script and AMD's
source are proved before every lease.

  hummingbird-xt-i2v  AMD's Hummingbird-XT i2v transformer and TurboVAED
                      decoder, the Wan 2.2 VAE and umT5 encoder from Wan-AI's
                      Diffusers safetensors, the umT5 tokenizer; 512x512, 81
                      frames at 16 fps, a first frame required, delivered as
                      H.264 MP4; a 31.0 GiB device memory lease
  flux2-klein-4b      Black Forest Labs' FLUX.2 [klein] 4B transformer and VAE
                      and the Qwen3 4B text encoder as Comfy-Org's single
                      files, BFL's Qwen3 tokenizer; 512x512 in four steps, one
                      image per request, delivered as PNG; an 18.5 GiB lease
                      alone, and beside hummingbird-xt-i2v the two sets'
                      40.9 GiB (BF16 decode) or 44.4 GiB (F32) together, the
                      second set loaded holding the difference

Usage:
  pytorchd setup inputs --python-home DIR --wheels DIR... --source REPO
  pytorchd setup install --python-home DIR --wheels DIR... [--root DIR]
                 [--state DIR] [--source DIR] [--no-compile]
  pytorchd setup remove [--root DIR] [--expected-root-key KEY]
  pytorchd serve --endpoint NAME [--python PATH] [--root DIR] [--source DIR]
                 [--video-model SET [--video-model-option NAME=VALUE]...]
                 [--image-model SET] [--resource-endpoint PATH]
                 [--resource NAME] [--idle DURATION] [--wait DURATION]
                 [--ready DURATION] [--video-time-limit DURATION]
                 [--image-time-limit DURATION] [--memory-limit BYTES]
                 [--runtime-program PATH]... [--state DIR]
  pytorchd engine check [--set SET] [--python PATH] [--root DIR] [--source DIR]
                 [--state DIR]
  pytorchd models install [--set SET] --from DIR... [--home DIR]
                 [--models-root DIR] [--job-store DIR] [--state DIR]
  pytorchd models check [--set SET] [--state DIR]
  pytorchd models protect [--models-root DIR] [--home DIR] [--state DIR]
                 [--job-store DIR] [--dry-run]
  pytorchd gpu list [--state DIR]
  pytorchd gpu clear (KEY | --all) [--state DIR]
  pytorchd help

Commands:
  setup inputs    print the enginefiles manifest of reviewed package inputs;
                  source is the Hummingbird repository, including its infer/
                  and notices. Creates no component and starts no process.
  setup install   install or update the engine from local files: copy the
                  base interpreter from --python-home and prove it by its
                  pins, make the venv without pip, install the lean lock's
                  wheels from the --wheels folders with every file checked
                  against its RECORD, and create the sandbox's profile and
                  grants, each recorded first; refused while the engine runs
  setup remove    remove the sandbox's grants and profile, proving every
                  access list is as it was, then the engine folder; the
                  provider's state and the model files stay
  serve           answer video@1 for --video-model and image@1 for
                  --image-model on the identity-bound local endpoint NAME, for
                  the designated runtime program, until interrupted; with both
                  the two sets share one engine process
  engine check    prove the interpreter, the engine script and SET's source as
                  a load would; exit 0 when serve would start it
  models install  find each pinned file of SET in each --from folder, prove
                  it where it lies and record where; nothing is copied
  models check    prove the recorded files where they lie
  models protect  make each recorded set folder under the models root a
                  model folder of this account; every other folder is listed
                  and left alone
  gpu list        print the requests paused after a graphics driver reset
  gpu clear       let the request KEY (or every paused request with --all) run
                  again
  help            print this help (also -h, --help, or no arguments)

Options:
  --endpoint NAME          local endpoint name
  --runtime-program P      absolute runtime executable permitted to call;
                           repeatable; default the installed runtime beside
                           this provider
  --python PATH            the reviewed venv's absolute python.exe; default
                           $PYTORCH_PYTHON, else the installed engine's
  --root DIR               the engine Setup installed, its sandbox and venv;
                           default %LOCALAPPDATA%\OpenAbstractions\engines\pytorch
  --python-home DIR        setup install: python.org's CPython 3.12.10 x64
                           folder the base interpreter is copied from
  --wheels DIR             setup install: a folder holding the lean lock's
                           wheels, searched with its subfolders; repeatable
  --no-compile             setup install: write no bytecode into the venv
  --source DIR             AMD's Hummingbird-XT infer/ folder at the reviewed
                           commit, for hummingbird-xt-i2v; default
                           $PYTORCH_SOURCE
  --video-model SET        serve video@1 for SET (hummingbird-xt-i2v)
  --video-model-option N=V one option of the video set; repeatable.
                           hummingbird-xt-i2v takes decode-dtype=f32 (the
                           default, AMD's) or bf16
  --image-model SET        serve image@1 for SET (flux2-klein-4b)
  --resource-endpoint PATH abstraction.resource/leases@1's platform endpoint;
                           default $ABSTRACTION_RESOURCE_ENDPOINT, or the
                           installed runtime's
  --resource NAME          the card the sets load into; default card:0
  --idle DURATION          unload a set nothing has asked for; default 5m
  --wait DURATION          how long holders have to yield for one load;
                           default 1m
  --ready DURATION         how long a set has to load; default 15m
  --video-time-limit D     how long one video may generate; default 30m
  --image-time-limit D     how long one image may generate; default 10m
  --memory-limit N         the engine job's per-process memory limit in bytes;
                           default the sets' measured commit with its margin on
                           Windows (43 GiB video, 25 GiB image, 55 GiB both)
  --state DIR              this provider's private state; default
                           <user cache>\OpenAbstractions\providers\pytorch
  --set SET                engine check and models: the reviewed PyTorch
                           model set; default hummingbird-xt-i2v
  --from DIR               models install: a folder that may hold the set's
                           files; repeatable, at least one
  --home DIR               models: read this directory as the user's home
  --models-root DIR        models: OA's model folder; default <home>\OA-models
  --job-store DIR          models: the runtime's job store
  --dry-run                models protect: report and write nothing

GPU faults: serve reads the Windows driver-reset and standby records and the
engine's own device-lost line. On a reset it stops the engine and releases
its leases, fails the running request gpu:reset and pauses an identical
request; in Modern Standby no new load starts. It never changes the Windows
TDR settings (resource CONTRACT.md RES-G7).

Rights: this provider's program needs abstraction.resource/hold on the card;
the runtime writes that rule for a declared provider that names card:0.
`

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr, os.Getenv))
}

func run(args []string, stdout, stderr io.Writer, getenv func(string) string) int {
	if len(args) == 0 {
		if _, err := io.WriteString(stdout, usage); err != nil {
			return 1
		}
		return 0
	}
	switch args[0] {
	case "help", "-h", "--help", "-help":
		if _, err := io.WriteString(stdout, usage); err != nil {
			return 1
		}
		return 0
	case "setup":
		return runSetup(args[1:], stdout, stderr)
	case "serve":
		return runServe(args[1:], stdout, stderr, getenv)
	case "engine":
		return runEngine(args[1:], stdout, stderr, getenv)
	case "models":
		return runModels(args[1:], stdout, stderr, getenv)
	case "gpu":
		return quarantine.Run("pytorchd", args[1:], stateFolder, usage, stdout, stderr)
	}
	//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
	fmt.Fprintf(stderr, "pytorchd: unknown command %q\n\n%s", args[0], usage)
	return 2
}

func wantsHelp(args []string) bool {
	for _, a := range args {
		if a == "-h" || a == "--help" || a == "-help" {
			return true
		}
	}
	return false
}

// repeated collects one option given more than once.
type repeated []string

func (r *repeated) String() string     { return strings.Join(*r, ",") }
func (r *repeated) Set(v string) error { *r = append(*r, v); return nil }

// videoInputLimit bounds one first frame the runtime stages in the
// provider's content store.
const videoInputLimit = 20 << 20

func runServe(args []string, stdout, stderr io.Writer, getenv func(string) string) int {
	if wantsHelp(args) {
		if _, err := io.WriteString(stdout, usage); err != nil {
			return 1
		}
		return 0
	}
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	//unchecked: usage text on a help/error path; a failed write to stderr has nothing further to report
	fs.Usage = func() { fmt.Fprint(stderr, usage) }
	endpoint := fs.String("endpoint", "", "")
	var runtimePrograms, videoOptions repeated
	fs.Var(&runtimePrograms, "runtime-program", "")
	python := fs.String("python", getenv("PYTORCH_PYTHON"), "")
	rootDir := fs.String("root", "", "")
	sourceDir := fs.String("source", getenv("PYTORCH_SOURCE"), "")
	videoModel := fs.String("video-model", "", "")
	fs.Var(&videoOptions, "video-model-option", "")
	imageModel := fs.String("image-model", "", "")
	resourceEndpoint := fs.String("resource-endpoint", getenv("ABSTRACTION_RESOURCE_ENDPOINT"), "")
	resourceName := fs.String("resource", torch.DefaultResource, "")
	idle := fs.Duration("idle", torch.DefaultIdle, "")
	wait := fs.Duration("wait", torch.DefaultWait, "")
	ready := fs.Duration("ready", torch.DefaultVideoReady, "")
	videoTimeLimit := fs.Duration("video-time-limit", torch.DefaultVideoTimeLimit, "")
	imageTimeLimit := fs.Duration("image-time-limit", torch.DefaultImageTimeLimit, "")
	memory := fs.Uint64("memory-limit", 0, "")
	stateDir := fs.String("state", "", "")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	fail := func(code int, format string, a ...any) int {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stderr, "pytorchd serve: %s\n", fmt.Sprintf(format, a...))
		return code
	}
	memoryGiven := false
	fs.Visit(func(f *flag.Flag) { memoryGiven = memoryGiven || f.Name == "memory-limit" })
	if *endpoint == "" {
		return fail(2, "--endpoint is required")
	}
	if *videoModel == "" && *imageModel == "" {
		return fail(2, "at least one --video-model or --image-model is required")
	}
	if fs.NArg() != 0 {
		return fail(2, "unexpected arguments %q", fs.Args())
	}
	var videoSet, imageSet torch.TorchModelSet
	if *videoModel != "" {
		set, ok := torch.TorchModelSetNamed(*videoModel)
		if !ok || set.Profile != "video" {
			return fail(2, "no reviewed PyTorch video set is named %q", *videoModel)
		}
		videoSet = set
	} else if len(videoOptions) > 0 {
		return fail(2, "--video-model-option belongs to --video-model")
	}
	if *imageModel != "" {
		set, ok := torch.TorchModelSetNamed(*imageModel)
		if !ok || set.Profile != "image" {
			return fail(2, "no reviewed PyTorch image set is named %q", *imageModel)
		}
		imageSet = set
	}
	videoChoices, err := torch.ParseTorchOptions(videoSet, videoOptions)
	if err != nil && *videoModel != "" {
		return fail(2, "%v", err)
	}
	programs, err := runtimeprog.Trusted(runtimePrograms)
	if err != nil {
		return fail(2, "%v", err)
	}
	state, err := stateFolder(*stateDir)
	if err != nil {
		return fail(2, "%v", err)
	}
	component, err := componentAt(*rootDir)
	if err != nil {
		return fail(2, "%v", err)
	}
	if *python == "" {
		*python = component.Python()
	}
	leaseEndpoint := *resourceEndpoint
	if leaseEndpoint == "" {
		leaseEndpoint = resource.DefaultEndpoint()
	}
	report := func(err error) {
		//unchecked: callback has no return path; a failed diagnostic cannot turn a failed load into success
		fmt.Fprintf(stderr, "pytorchd: %v\n", err)
	}
	leases := resource.New(leaseEndpoint)
	gpu, err := newGPUGuard(state, leases, stderr, report)
	if err != nil {
		return fail(1, "%v", err)
	}
	storeName := "pytorch-video"
	if *videoModel == "" {
		storeName = "pytorch-image"
	}
	content, err := providerhost.NewDesignatedContent(storeName, filepath.Join(state, "content"),
		filepath.Join(state, "content-uploads.json"), programs, videoInputLimit)
	if err != nil {
		return fail(1, "%v", err)
	}
	engine := func(set torch.TorchModelSet, options map[string]string, limit uint64) *torch.TorchEngine {
		if memoryGiven {
			limit = *memory
		} else if runtime.GOOS != "windows" {
			limit = 0
		}
		return &torch.TorchEngine{Python: *python, Source: *sourceDir, Set: set, Options: options,
			Places: torch.TorchModelPlaces(state, set), ModelRecord: torch.TorchModelRecord(state, set),
			State: filepath.Join(state, "torch"), Ready: *ready, Log: filepath.Join(state, "logs", "torch.log"), MemoryLimit: limit,
			Sandbox: sandboxManifest(component)}
	}
	options := torch.Options{Leases: leases, GPU: gpu, Resource: *resourceName, Idle: *idle, Wait: *wait,
		OnError: report, OnEngineExit: engineproc.Report(slog.New(logging.LegacyDefault("pytorchd")), stderr, "pytorchd")}
	var imagePlace *torch.TorchSetPlace
	if *imageModel != "" {
		imagePlace = &torch.TorchSetPlace{Set: imageSet, Places: torch.TorchModelPlaces(state, imageSet),
			ModelRecord: torch.TorchModelRecord(state, imageSet)}
		options.Image = &torch.ImageOptions{Model: *imageModel, TimeLimit: *imageTimeLimit}
	}
	if *videoModel != "" {
		// The job's per-process limit is the set's measured peak commit with
		// its margin, or both sets' together in one process (PYTORCH-T5,
		// PYTORCH-T10).
		limit := videoSet.Commit
		if imagePlace != nil {
			limit = torch.TorchCommitTogether(videoSet, videoChoices, imageSet)
		}
		e := engine(videoSet, videoChoices, limit)
		e.Beside = imagePlace
		options.Video = &torch.VideoOptions{Model: *videoModel, Torch: e, Store: content.Store, TimeLimit: *videoTimeLimit}
	} else {
		e := engine(imageSet, nil, imageSet.Commit)
		e.Source = ""
		options.Image.Torch = e
	}
	h, err := torch.New(options)
	if err != nil {
		//unchecked: the construction failure is reported below
		content.Component.Close()
		return fail(2, "%v", err)
	}
	defer h.Close()
	providerConfig := torch.ProviderConfig(h, report)
	if options.Image != nil {
		// image@1 results are the caller's: the provider publishes each
		// through its own content writer (PYTORCH-T11).
		providerConfig.PrepareContentWrite, err = content.OutputWriter(listen.Endpoint(*endpoint), torch.ImageOutputLimit)
		if err != nil {
			return fail(1, "%v", err)
		}
	}
	provider, err := inference.New(providerConfig)
	if err != nil {
		return fail(1, "%v", err)
	}
	defer provider.Close()
	served, err := torch.Listen(listen.Endpoint(*endpoint), provider, programs...)
	if err != nil {
		return fail(1, "%v", err)
	}
	// video@1 and the content store the runtime stages first frames in and
	// reads results from share this endpoint; image@1 publishes into it.
	served.Content = content.Component
	if options.Video != nil {
		served.Video = h.VideoServer()
	}
	served.OnError = func(err error) {
		//unchecked: callback has no return path; a failed diagnostic cannot turn a failed call into success
		fmt.Fprintf(stderr, "pytorchd serve: call: %v\n", err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	done := make(chan struct{})
	go func() { defer close(done); h.Run(ctx) }()
	go gpu.Run(ctx)
	defer gpu.Flush()
	var sets []string
	if *videoModel != "" {
		sets = append(sets, *videoModel+" (video@1)")
	}
	if *imageModel != "" {
		sets = append(sets, *imageModel+" (image@1)")
	}
	//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
	fmt.Fprintf(stdout, "pytorchd serving %s on %s; the engine starts on demand\n", strings.Join(sets, ", "), listen.Endpoint(*endpoint))
	err = served.Serve(ctx)
	stop()
	h.Close()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
	}
	if err != nil {
		return fail(1, "%v", err)
	}
	return 0
}

// newGPUGuard is serve's GPU fault handling: the resource kit's provider
// composition (gpufault.ForProvider) with the wake lease through leases@1,
// as every card provider takes it, and OA logging.
func newGPUGuard(state string, leases holder.Leases, stderr io.Writer, report func(error)) (*gpufault.Guard, error) {
	return gpufault.ForProvider(gpufault.ProviderOptions{Name: "pytorchd", State: state,
		Wake: gpufault.LeaseWake(leases, 0, report), Logger: slog.New(logging.LegacyDefault("pytorchd")),
		Stderr: stderr, Report: report})
}

// runEngine proves the reviewed interpreter, the engine script and a set's
// source as a load does. Nothing is installed: Setup builds the venv from
// the stack review's lock.
func runEngine(args []string, stdout, stderr io.Writer, getenv func(string) string) int {
	if len(args) == 0 || wantsHelp(args) || args[0] == "help" {
		if _, err := io.WriteString(stdout, usage); err != nil {
			return 1
		}
		return 0
	}
	fs := flag.NewFlagSet("engine "+args[0], flag.ContinueOnError)
	fs.SetOutput(stderr)
	//unchecked: usage text on a help/error path; a failed write to stderr has nothing further to report
	fs.Usage = func() { fmt.Fprint(stderr, usage) }
	setName := fs.String("set", torch.HummingbirdXTI2VTorch.Name, "")
	python := fs.String("python", getenv("PYTORCH_PYTHON"), "")
	rootDir := fs.String("root", "", "")
	sourceDir := fs.String("source", "", "")
	stateDir := fs.String("state", "", "")
	if err := fs.Parse(args[1:]); err != nil {
		return 2
	}
	fail := func(code int, format string, a ...any) int {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stderr, "pytorchd engine %s: %s\n", args[0], fmt.Sprintf(format, a...))
		return code
	}
	if args[0] != "check" {
		return fail(2, "engine takes check; pytorchd setup install builds the venv from the stack review's lock")
	}
	if fs.NArg() != 0 {
		return fail(2, "unexpected arguments %q", fs.Args())
	}
	component, err := componentAt(*rootDir)
	if err != nil {
		return fail(2, "%v", err)
	}
	if *python == "" {
		*python = component.Python()
	}
	set, ok := torch.TorchModelSetNamed(*setName)
	if !ok {
		return fail(2, "no reviewed PyTorch model set is named %q", *setName)
	}
	needsSource := len(set.Source) > 0
	if *python == "" || !filepath.IsAbs(*python) || needsSource && (*sourceDir == "" || !filepath.IsAbs(*sourceDir)) || !needsSource && *sourceDir != "" {
		return fail(2, "--python names the venv's absolute python.exe, and --source the absolute folder of AMD's Hummingbird-XT infer/ source for a set that runs it")
	}
	state, err := stateFolder(*stateDir)
	if err != nil {
		return fail(2, "%v", err)
	}
	if err := torch.CheckTorchEngine(context.Background(), *python, *sourceDir, filepath.Join(state, "torch"), sandboxManifest(component), set); err != nil {
		return fail(1, "%v", err)
	}
	if needsSource {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stdout, "%s, the OA PyTorch engine and the %s source are the reviewed ones\n", *python, set.Name)
	} else {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stdout, "%s and the OA PyTorch engine for %s are the reviewed ones\n", *python, set.Name)
	}
	return 0
}

// componentAt is the engine Setup installs at dir, or at the default root.
func componentAt(dir string) (torch.Component, error) {
	if dir == "" {
		root, err := torch.DefaultComponentRoot()
		if err != nil {
			return torch.Component{}, err
		}
		dir = root
	}
	if !filepath.IsAbs(dir) {
		return torch.Component{}, fmt.Errorf("--root %s is not an absolute path", dir)
	}
	return torch.Component{Root: filepath.Clean(dir)}, nil
}

// sandboxManifest is the manifest of the component's sandbox, which every
// launch on Windows checks and starts in; elsewhere no AppContainer exists
// and no stack is pinned (PYTORCH-T12).
func sandboxManifest(c torch.Component) string {
	if runtime.GOOS == "windows" {
		return c.Manifest()
	}
	return ""
}

// runSetup installs, updates or removes the PyTorch engine (TODO D5a).
var componentInputRoster = torch.ComponentInputs // read-only command seam
var componentRemoval = torch.RemoveComponent     // ordinary removal dispatch fixture

func runSetup(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 || wantsHelp(args) || args[0] == "help" {
		if _, err := io.WriteString(stdout, usage); err != nil {
			return 1
		}
		return 0
	}
	fs := flag.NewFlagSet("setup "+args[0], flag.ContinueOnError)
	fs.SetOutput(stderr)
	//unchecked: usage text on a help/error path; a failed write to stderr has nothing further to report
	fs.Usage = func() { fmt.Fprint(stderr, usage) }
	rootDir := fs.String("root", "", "")
	expectedRootKey := fs.String("expected-root-key", "", "")
	pythonHome := fs.String("python-home", "", "")
	var wheels repeated
	fs.Var(&wheels, "wheels", "")
	stateDir := fs.String("state", "", "")
	sourceDir := fs.String("source", "", "")
	noCompile := fs.Bool("no-compile", false, "")
	if err := fs.Parse(args[1:]); err != nil {
		return 2
	}
	fail := func(code int, format string, a ...any) int {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stderr, "pytorchd setup %s: %s\n", args[0], fmt.Sprintf(format, a...))
		return code
	}
	if fs.NArg() != 0 {
		return fail(2, "unexpected arguments %q", fs.Args())
	}
	if args[0] == "inputs" {
		forbidden := false
		fs.Visit(func(option *flag.Flag) {
			if option.Name == "root" || option.Name == "state" || option.Name == "no-compile" || option.Name == "expected-root-key" {
				forbidden = true
			}
		})
		if forbidden {
			return fail(2, "inputs takes --python-home, --wheels and --source alone")
		}
		manifest, err := componentInputRoster(context.Background(), *pythonHome, wheels, *sourceDir)
		if err != nil {
			return fail(1, "%v", err)
		}
		if err := json.NewEncoder(stdout).Encode(manifest); err != nil {
			return fail(1, "%v", err)
		}
		return 0
	}
	component, err := componentAt(*rootDir)
	if err != nil {
		return fail(2, "%v", err)
	}
	switch args[0] {
	case "install":
		forbidden := false
		fs.Visit(func(option *flag.Flag) { forbidden = forbidden || option.Name == "expected-root-key" })
		if forbidden {
			return fail(2, "install does not take --expected-root-key")
		}
		if *pythonHome == "" || len(wheels) == 0 {
			return fail(2, "pass --python-home DIR, python.org's CPython 3.12.10 x64 folder, and --wheels DIR for the lean lock's wheels")
		}
		state, err := stateFolder(*stateDir)
		if err != nil {
			return fail(2, "%v", err)
		}
		err = torch.InstallComponent(context.Background(), torch.ComponentInstall{Component: component, PythonHome: *pythonHome,
			Wheels: wheels, State: filepath.Join(state, "torch"), Source: *sourceDir, Compile: !*noCompile, Out: stdout})
		if err != nil {
			return fail(1, "%v", err)
		}
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stdout, "installed; run pytorchd engine check, then pytorchd models install for each set\n")
		return 0
	case "remove":
		if *pythonHome != "" || len(wheels) > 0 || *stateDir != "" || *sourceDir != "" || *noCompile {
			return fail(2, "remove takes --root and an optional --expected-root-key")
		}
		component.ExpectedRootKey = *expectedRootKey
		if _, err := componentRemoval(component, stdout); err != nil {
			return fail(1, "%v", err)
		}
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stdout, "removed %s\n", component.Root)
		return 0
	}
	return fail(2, "setup takes inputs, install or remove")
}

// modelsHostFlags are the `openabstractions models host` flags that declare
// set: an image set by --image-model, a video set by --video-kind torch
// --video-model (live walkthrough 2026-10-02, D13).
func modelsHostFlags(set torch.TorchModelSet) string {
	if set.Profile == "image" {
		return "--image-model " + set.Name
	}
	return "--video-kind torch --video-model " + set.Name
}

// runModels records where the pinned files of a PyTorch set lie and proves
// them there (KIT-M1); nothing is copied.
func runModels(args []string, stdout, stderr io.Writer, getenv func(string) string) int {
	if len(args) == 0 || wantsHelp(args) || args[0] == "help" {
		if _, err := io.WriteString(stdout, usage); err != nil {
			return 1
		}
		return 0
	}
	fs := flag.NewFlagSet("models "+args[0], flag.ContinueOnError)
	fs.SetOutput(stderr)
	//unchecked: usage text on a help/error path; a failed write to stderr has nothing further to report
	fs.Usage = func() { fmt.Fprint(stderr, usage) }
	setName := fs.String("set", torch.HummingbirdXTI2VTorch.Name, "")
	var from repeated
	fs.Var(&from, "from", "")
	home := fs.String("home", "", "")
	stateDir := fs.String("state", "", "")
	modelsRoot := fs.String("models-root", "", "")
	jobStore := fs.String("job-store", modelplace.DefaultJobStore(getenv), "")
	dryRun := fs.Bool("dry-run", false, "")
	if err := fs.Parse(args[1:]); err != nil {
		return 2
	}
	fail := func(code int, format string, a ...any) int {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stderr, "pytorchd models %s: %s\n", args[0], fmt.Sprintf(format, a...))
		return code
	}
	if fs.NArg() != 0 {
		return fail(2, "unexpected arguments %q", fs.Args())
	}
	homeDir := *home
	if homeDir == "" {
		homeDir = userHome(getenv)
	}
	root := *modelsRoot
	if root == "" && homeDir != "" {
		root = modelplace.ModelsRoot(homeDir)
	}
	if root != "" && !filepath.IsAbs(root) {
		return fail(2, "--models-root names an absolute folder")
	}
	var linkRoots []string
	if *jobStore != "" {
		if !filepath.IsAbs(*jobStore) {
			return fail(2, "--job-store names an absolute folder")
		}
		linkRoots = append(linkRoots, filepath.Clean(*jobStore))
	}
	if args[0] == "protect" {
		if root == "" {
			return fail(2, "no models root; pass --models-root DIR or --home DIR")
		}
		state, err := statePath(*stateDir)
		if err != nil {
			return fail(2, "%v", err)
		}
		recorded, err := torch.RecordedSetFolders(filepath.Clean(root), state)
		if err != nil {
			return fail(1, "%v", err)
		}
		failed, err := modelplace.ProtectFolders(root, recorded, linkRoots, *dryRun, stdout)
		if err != nil {
			return fail(1, "%v", err)
		}
		if failed > 0 {
			return fail(1, "%d set folders were not protected", failed)
		}
		return 0
	}
	if *dryRun {
		return fail(2, "--dry-run belongs to models protect")
	}
	set, ok := torch.TorchModelSetNamed(*setName)
	if !ok {
		return fail(2, "no reviewed PyTorch model set is named %q", *setName)
	}
	state, err := stateFolder(*stateDir)
	if err != nil {
		return fail(2, "%v", err)
	}
	places, record := torch.TorchModelPlaces(state, set), torch.TorchModelRecord(state, set)
	switch args[0] {
	case "install":
		// The --from folders alone: the provider reads no other
		// application's store itself, and a store search through the
		// runtime's storage service is TODO D5d.
		if len(from) == 0 {
			return fail(2, "pass --from DIR, a folder that holds the set's files")
		}
		searches, err := modelplace.Searches(from, nil)
		if err != nil {
			return fail(2, "%v", err)
		}
		found, err := torch.PlaceTorchModels(context.Background(), set, searches, places)
		if err != nil {
			return fail(1, "%v", err)
		}
		// Each of OA's set folders under the models root that now holds a
		// recorded file becomes a model folder, so its files pass the
		// trusted-in-place check (identity ID-D3).
		err = modelplace.AdoptSetFolders(root, linkRoots, set.Files, found, func(dir string) {
			//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
			fmt.Fprintf(stdout, "%s: OA's model folder, protected\n", dir)
		})
		if err != nil {
			return fail(1, "%v", err)
		}
		if err := torch.CheckTorchModels(context.Background(), enginefiles.NewVerifier(enginefiles.Options{}), set, places, record); err != nil {
			return fail(1, "%v", err)
		}
		for _, f := range set.Files {
			//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
			fmt.Fprintf(stdout, "%s: %s\n", f.Name, found[f.Name])
		}
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stdout, "recorded the %s files where they lie; nothing was copied; declare it with openabstractions models host %s\n",
			set.Name, modelsHostFlags(set))
		return 0
	case "check":
		if len(from) > 0 {
			return fail(2, "--from belongs to models install")
		}
		if err := torch.CheckTorchModels(context.Background(), enginefiles.NewVerifier(enginefiles.Options{}), set, places, record); err != nil {
			return fail(1, "%v", err)
		}
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stdout, "the %s files are the pinned ones where they were recorded\n", set.Name)
		return 0
	}
	return fail(2, "unknown command; see pytorchd help")
}

func userHome(getenv func(string) string) string {
	if runtime.GOOS == "windows" {
		if h := getenv("USERPROFILE"); h != "" {
			return h
		}
	} else if h := getenv("HOME"); h != "" {
		return h
	}
	//unchecked: a failure just leaves h empty, which the caller checks and reports
	h, _ := os.UserHomeDir()
	return h
}

// stateFolder is this provider's private state: dir when it is given, or
// <user cache>/OpenAbstractions/providers/pytorch. It is made when absent.
func stateFolder(dir string) (string, error) {
	dir, err := statePath(dir)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	return dir, nil
}

// statePath is the state folder stateFolder names, without making it.
func statePath(dir string) (string, error) {
	if dir == "" {
		cache, err := os.UserCacheDir()
		if err != nil {
			return "", fmt.Errorf("no state folder; pass --state: %w", err)
		}
		dir = filepath.Join(cache, "OpenAbstractions", "providers", "pytorch")
	}
	if !filepath.IsAbs(dir) {
		return "", fmt.Errorf("--state %s is not an absolute path", dir)
	}
	return filepath.Clean(dir), nil
}
