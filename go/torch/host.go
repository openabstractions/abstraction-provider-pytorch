// Package torch is the OA PyTorch engine provider: one engine process, the
// provider's own Python script on the reviewed PyTorch for ROCm venv, serving
// a video set, an image set, or both in one process, each set under its own
// lease of the card through the resource kit's holder and the GPU fault
// Guard (CONTRACT.md PYTORCH-T1..PYTORCH-T11). OA supplies this engine
// because no engine on the machine serves these models (VISION 2026-10-02).
package torch

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"runtime"
	"time"

	identity "github.com/openabstractions/abstraction-identity"
	inference "github.com/openabstractions/abstraction-inference/go"
	"github.com/openabstractions/abstraction-inference/go/providerhost/devicelost"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginelaunch"
	"github.com/openabstractions/abstraction-provider-kit/go/engineproc"
	"github.com/openabstractions/abstraction-provider-kit/go/refusal"
	"github.com/openabstractions/abstraction-provider-kit/go/videoops"
	res "github.com/openabstractions/abstraction-resource/go/client"
	"github.com/openabstractions/abstraction-resource/go/gpufault"
	"github.com/openabstractions/abstraction-resource/go/holder"
	router "github.com/openabstractions/abstraction-router/go"
	storage "github.com/openabstractions/abstraction-storage/go"
)

// Options are the provider's lease service and card, its holder timings, its
// GPU fault handling, and the video and image sets it serves.
type Options struct {
	Leases   holder.Leases
	Resource string
	Idle     time.Duration
	Renew    time.Duration
	Wait     time.Duration
	Now      func() time.Time
	OnError  func(error)
	// OnEngineExit, when set, receives the engine's exit record once its
	// process tree has ended (KIT-P3).
	OnEngineExit func(EngineExit)
	// Video, when set, serves video@1 for its set on the engine.
	Video *VideoOptions
	// Image, when set, serves image@1 for its set, in the video set's engine
	// process when there is one (PYTORCH-T10, T11).
	Image *ImageOptions
	// GPU is the provider's GPU fault handling: asked before every load,
	// it stops the engine on a driver reset and hears the engine's own
	// device-lost line (resource RES-G1..RES-G7).
	GPU *gpufault.Guard
}

// VideoOptions is the video set: the model name callers ask for, the
// engine and its set, the provider's content store and the video bounds.
type VideoOptions struct {
	// Model is the name the router lists; empty reads the set's name.
	Model string
	// Torch is the engine and its set.
	Torch *TorchEngine
	// Store is the provider's content store: the runtime stages first
	// frames in it and reads results from it.
	Store     *storage.ContentStore
	TimeLimit time.Duration
	Retention time.Duration
}

// DefaultResource is the card the engine loads into.
const DefaultResource = "card:0"

const (
	DefaultIdle  = 5 * time.Minute
	DefaultRenew = 8 * time.Second
	DefaultWait  = 60 * time.Second
)

// The words a load or a request is refused with.
const (
	NoEngine       = refusal.NoEngine
	Unavailable    = refusal.Unavailable
	NotPermitted   = holder.WordNotPermitted
	Insufficient   = holder.WordInsufficient
	HoldersRefused = holder.WordHoldersRefused
	LeaseEnded     = holder.WordEnded
	// EngineNetworkSocket is an engine whose contained tree holds a network
	// socket (inference INF-EL2); the PyTorch engine speaks over its pipes.
	EngineNetworkSocket = "engine:network-socket"
)

// Host owns the PyTorch engine process and its sets' leases.
type Host struct {
	options   Options
	resources *holder.Holder
	video     *videoHost
	torch     *torchShared
	image     *torchImage
	// watch is how often a running engine's sockets are checked again.
	watch time.Duration
	// afterFailedCheck, set by tests only, runs between a failed running
	// socket check and the stop it leads to.
	afterFailedCheck func()
}

// New composes the provider from options; it starts no engine.
func New(options Options) (*Host, error) {
	if options.Leases == nil {
		return nil, errors.New("pytorch: the lease service is required")
	}
	if options.Video == nil && options.Image == nil {
		return nil, errors.New("pytorch: a video set or an image set is required")
	}
	if options.Resource == "" {
		options.Resource = DefaultResource
	}
	if options.Idle <= 0 {
		options.Idle = DefaultIdle
	}
	if options.Renew <= 0 {
		options.Renew = DefaultRenew
	}
	if options.Wait <= 0 {
		options.Wait = DefaultWait
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	files := enginefiles.NewVerifier(enginefiles.Options{})
	normalTorch := func(t TorchEngine) *TorchEngine {
		if t.runtime.Release == "" {
			//unchecked: a platform with no pin leaves the runtime zero, and every load is refused engine:none
			t.runtime, _ = pinnedTorchRuntime(runtime.GOOS, runtime.GOARCH)
		}
		if t.Files == nil {
			t.Files = files
		}
		t.report = options.OnError
		t.exited = options.OnEngineExit
		return &t
	}
	if v := options.Video; v != nil {
		if v.Torch == nil {
			return nil, errors.New("pytorch: the video set needs the PyTorch engine")
		}
		t := normalTorch(*v.Torch)
		if t.Set.Name == "" {
			t.Set = HummingbirdXTI2VTorch
		}
		if v.Model == "" {
			v.Model = t.Set.Name
		}
		v.Torch = t
	}
	if im := options.Image; im != nil {
		switch {
		case options.Video != nil:
			// The image set joins the video set's engine process.
			if im.Torch != nil || options.Video.Torch.Beside == nil {
				return nil, errors.New("pytorch: with a video set the image set is that engine's Beside")
			}
			if im.Model == "" {
				im.Model = options.Video.Torch.Beside.Set.Name
			}
		case im.Torch != nil:
			im.Torch = normalTorch(*im.Torch)
			if im.Model == "" {
				im.Model = im.Torch.Set.Name
			}
		default:
			return nil, errors.New("pytorch: the image set needs the PyTorch engine")
		}
		if options.Video != nil && options.Video.Model == im.Model {
			return nil, fmt.Errorf("pytorch: %q names both the video model and the image model", im.Model)
		}
	}
	var admit func(context.Context, string) error
	if gpu := options.GPU; gpu != nil {
		admit = func(ctx context.Context, _ string) error { return gpuRefusal(gpu.AdmitLoad(ctx)) }
		// The engine's device-lost line counts only once the engine exits
		// or fails its own readiness check twice (KIT-P4, resource RES-G1).
		hear := engineproc.DeviceLost(gpu, "pytorch", "oa-torch-engine", options.OnError)
		lost := func(p *Process, marker string) {
			hear(p.exited, func(ctx context.Context) bool {
				ok, err := p.ready(ctx)
				return err == nil && ok
			}, marker)
		}
		if options.Video != nil {
			options.Video.Torch.deviceLost = lost
		}
		if options.Image != nil && options.Image.Torch != nil {
			options.Image.Torch.deviceLost = lost
		}
	}
	var h *Host
	r, err := holder.New(holder.Options{
		Admit:  admit,
		Leases: options.Leases, Resource: options.Resource, Idle: options.Idle,
		Renew: options.Renew, Wait: options.Wait, Now: options.Now,
		// A set's component is in use while its own work runs, and, when
		// its lease is the process's own, while any set in that process
		// works (PYTORCH-T10).
		Active: func(name string) bool {
			return h != nil && h.torch != nil && h.torch.busy(name)
		},
		OnError: options.OnError,
	})
	if err != nil {
		return nil, err
	}
	h = &Host{options: options, resources: r, watch: socketWatch}
	if options.GPU != nil {
		// A reset stops the engine's process directly before the holder
		// stops every component (PYTORCH-T10).
		options.GPU.Attach(torchLoser{h})
	}
	if options.Video != nil {
		h.torch = newTorchShared(h, *options.Video.Torch)
	} else {
		h.torch = newTorchShared(h, *options.Image.Torch)
	}
	if options.Video != nil {
		if h.video, err = newVideoHost(h, *options.Video); err != nil {
			return nil, err
		}
		h.torch.active[h.video.opts.Model] = h.video.Active
	}
	if options.Image != nil {
		if h.image, err = newTorchImage(h, h.torch, *options.Image); err != nil {
			if h.video != nil {
				h.video.Close()
			}
			return nil, err
		}
		h.torch.active[h.image.model] = h.image.active
	}
	return h, nil
}

// ImageEngine is the image set and the router host it serves, for the
// inference provider's Config.ImageEngine and ImageEngineHost; both are nil
// without one (PYTORCH-T11).
func (h *Host) ImageEngine() (inference.ImageEngine, *router.Host) {
	if h.image == nil {
		return nil, nil
	}
	return h.image, h.image.route
}

// ServesImage reports whether name is the image set's model.
func (h *Host) ServesImage(name string) bool { return h.image != nil && h.image.model == name }

// VideoServer is this provider's video@1, for the service host's Video
// field; nil when no video set is served.
func (h *Host) VideoServer() *videoHost { return h.video }

func (h *Host) Residents() []string      { return h.resources.Residents() }
func (h *Host) Loaded(name string) bool  { _, ok := h.resources.Snapshot(name); return ok }
func (h *Host) Lease(name string) string { lease, _ := h.resources.Snapshot(name); return lease }
func (h *Host) Unload(name, why string)  { h.resources.Unload(name, why) }
func (h *Host) Run(ctx context.Context)  { h.resources.Run(ctx) }

// Close stops the video worker, then the engine, and releases the leases.
func (h *Host) Close() {
	if h.video != nil {
		h.video.Close()
	}
	h.resources.Close()
}

func (h *Host) yield(request res.YieldRequest) (res.YieldAnswer, string) {
	return h.resources.Yield(request)
}

func (h *Host) mapFailure(name string, amount int64, err error) error {
	return refusal.Load(h.options.Resource, name, amount, err)
}

// videoHost is the provider's video@1: the provider kit's operation table
// (videoops) running the video set on the engine.
type videoHost struct {
	*videoops.Table
	host *Host
	opts VideoOptions
	kind videoops.Kind
}

func newVideoHost(h *Host, opts VideoOptions) (*videoHost, error) {
	if opts.Store == nil {
		return nil, errors.New("pytorch: the video set needs a content store")
	}
	if opts.Model != opts.Torch.Set.Name {
		return nil, fmt.Errorf("pytorch: the video model %q is not the model set %q", opts.Model, opts.Torch.Set.Name)
	}
	if err := opts.Torch.Set.valid(); err != nil {
		return nil, fmt.Errorf("pytorch: %w", err)
	}
	if err := opts.Torch.Set.checkOptions(opts.Torch.Options); err != nil {
		return nil, fmt.Errorf("pytorch: %w", err)
	}
	table, err := videoops.New(videoops.Options{Model: opts.Model, Store: opts.Store, TimeLimit: opts.TimeLimit,
		Retention: opts.Retention, Holder: h.resources, GPU: h.options.GPU, Now: h.options.Now})
	if err != nil {
		return nil, fmt.Errorf("pytorch: %w", err)
	}
	v := &videoHost{Table: table, host: h, opts: opts}
	sp, _ := h.torch.engine.place(opts.Model)
	v.kind = &torchKind{v: v, shared: h.torch, sp: sp}
	table.Serve(v.kind)
	return v, nil
}

func (v *videoHost) active() bool { return v.Table.Active() }

// A Process is the running engine: the provider kit's supervised process
// (KIT-P1) and the readiness its protocol answers.
type Process struct {
	owned    *engineproc.Process
	id       int
	exited   <-chan struct{}
	contains identity.TreeContains
	// probe reports readiness: the engine's ready message, then a ping.
	probe func(context.Context, *http.Client) bool
}

func (p *Process) PID() int {
	if p == nil {
		return 0
	}
	return p.id
}

func (p *Process) stop(within time.Duration) {
	if p == nil || p.owned == nil {
		return
	}
	deadline := time.Now().Add(within)
	p.owned.Stop(within)
	// The kit confirms the tree has exited; this wrapper also waits for the
	// provider's post-exit file and sandbox cleanup before returning.
	if p.owned.Gone() {
		waitProcessCleanup(p.exited, deadline)
	}
}

// waitProcessCleanup spends only the stop budget left after tree exit.
func waitProcessCleanup(done <-chan struct{}, deadline time.Time) bool {
	select {
	case <-done:
		return true
	default:
	}
	left := time.Until(deadline)
	if left <= 0 {
		return false
	}
	timer := time.NewTimer(left)
	defer timer.Stop()
	select {
	case <-done:
		return true
	case <-timer.C:
		return false
	}
}

func (p *Process) ready(ctx context.Context) (bool, error) {
	if p.probe == nil {
		return false, nil
	}
	return p.probe(ctx, nil), nil
}

// startOwned starts spec through the provider kit (KIT-P1, KIT-P3).
func startOwned(spec ownedSpec) (*Process, error) {
	spec.Provider = "pytorch"
	owned, err := engineproc.Start(spec)
	if errors.Is(err, engineproc.ErrEnvironment) {
		return nil, ErrEngineEnvironment
	}
	if err != nil {
		return nil, err
	}
	return &Process{owned: owned, id: owned.PID(), exited: owned.Exited(), contains: owned.Contains()}, nil
}

// engineOutput is where the engine's output goes, read on the way for its
// device-lost line (KIT-P4).
func engineOutput(out io.Writer, form devicelost.LineForm, deviceLost func(*Process, string)) (io.Writer, *engineproc.LostLine[*Process]) {
	return engineproc.Output(out, form, deviceLost)
}

// The socket owner tables and how often a running engine's sockets are
// read again; tests replace them.
var (
	ownerSockets    = identity.Sockets
	ownerAllSockets = identity.AllSockets
	socketWatch     = 30 * time.Second
)

// ownedProcess is the engine's process as the holder owns it (KIT-P5): its
// readiness, then the rule that a pipe engine holds no network socket at
// all, at readiness and every watch while it runs (inference INF-EL2).
type ownedProcess struct {
	name    string
	process *Process
	ready   time.Duration
	host    *Host
}

func (o *ownedProcess) kit() *engineproc.Owned {
	h, p := o.host, o.process
	return &engineproc.Owned{Name: o.name, Process: p.owned, ReadyWithin: o.ready, Probe: p.ready,
		Sockets: func() error {
			return enginelaunch.CheckEngineSockets(ownerSockets, ownerAllSockets, p.contains, 0)
		},
		SocketWord: EngineNetworkSocket, Watch: h.watch,
		Resident: func() bool { return h.resources.PID(o.name) == p.PID() }, Report: h.options.OnError,
		AfterFailedCheck: h.afterFailedCheck}
}

func (o *ownedProcess) PID() int                        { return o.process.PID() }
func (o *ownedProcess) Ready(ctx context.Context) error { return o.kit().Ready(ctx) }
func (o *ownedProcess) Alive() bool {
	select {
	case <-o.process.exited:
		return false
	default:
		return true
	}
}
func (o *ownedProcess) Stop() bool {
	deadline := time.Now().Add(engineproc.StopWithin)
	stopped := o.kit().Stop()
	return stopped && waitProcessCleanup(o.process.exited, deadline)
}

// ErrEngineEnvironment reports an engine environment entry outside the
// allowlist (PYTORCH-T1; stack review condition 3).
var ErrEngineEnvironment = errors.New("pytorch: engine environment variable outside the allowlist")

// registryString reads the profile registry for the clean environment's
// builder; tests replace it.
var registryString enginelaunch.RegistryValue = enginelaunch.SystemRegistry
