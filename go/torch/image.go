package torch

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	"github.com/openabstractions/abstraction-inference/go/providerhost/rgbpng"
	"github.com/openabstractions/abstraction-inference/go/providerhost/sdcpp"
	"github.com/openabstractions/abstraction-resource/go/gpufault"
	router "github.com/openabstractions/abstraction-router/go"
)

// ImageOptions add the image engine kind: the OA PyTorch engine serving
// image@1 for one model set (PYTORCH-T10, T11).
type ImageOptions struct {
	// Model is the name the router lists and callers ask for; empty reads
	// the set's name, and any other name is refused.
	Model string
	// Torch is the engine when no PyTorch video set runs; its Set is the
	// image set. With a PyTorch video kind, the image set is that engine's
	// Beside, the two share one process, and Torch stays nil.
	Torch *TorchEngine
	// TimeLimit bounds one image once the engine took it; zero reads
	// DefaultImageTimeLimit.
	TimeLimit time.Duration
}

const (
	// DefaultImageTimeLimit bounds one image: a start image took 6.0 to 9.4
	// s warm on 2026-10-02, and a cold load 16.3 s besides.
	DefaultImageTimeLimit = 10 * time.Minute
	// torchImageCancelWithin bounds how long a cancelled image waits for the
	// engine to stop between two steps before its process is stopped.
	torchImageCancelWithin = 30 * time.Second
	// imageRouteName is the router host of the image engine kind.
	imageRouteName = "pytorch-image"
)

// Image failure reasons a reply carries. None carries engine text.
const (
	imageEngineFailed  = "engine:failed"
	imageEngineResult  = "engine:result"
	imageEngineTimeout = "engine:timeout"
	imageEngineLost    = "engine:lost"
	imageEngineRefused = "engine:refused"
)

// torchImage serves image@1 on the OA PyTorch engine: one image per
// request, from the prompt alone, at a size the set measured
// (PYTORCH-T11). It is the inference provider's explicitly composed image
// engine (inference Config.ImageEngine); the provider keeps admission,
// bounds and the caller's publication.
type torchImage struct {
	host      *Host
	shared    *torchShared
	sp        TorchSetPlace
	model     string
	timeLimit time.Duration
	route     *router.Host

	mu      sync.Mutex
	waiting int
	// cancelWithin replaces torchImageCancelWithin; only tests set it.
	cancelWithin time.Duration
}

func newTorchImage(h *Host, shared *torchShared, opts ImageOptions) (*torchImage, error) {
	sp, ok := shared.engine.place(opts.Model)
	if !ok || sp.Set.Profile != torchProfileImage {
		return nil, fmt.Errorf("pytorch: the image model %q is no image set of the PyTorch engine", opts.Model)
	}
	if opts.TimeLimit <= 0 {
		opts.TimeLimit = DefaultImageTimeLimit
	}
	route, err := router.NewImageEngine(imageRouteName, []string{opts.Model})
	if err != nil {
		return nil, err
	}
	return &torchImage{host: h, shared: shared, sp: sp, model: opts.Model, timeLimit: opts.TimeLimit, route: route}, nil
}

// active reports a request waiting for or running on the engine.
func (k *torchImage) active() bool {
	k.mu.Lock()
	defer k.mu.Unlock()
	return k.waiting > 0
}

// size is the request's size: the preset's for auto, else WIDTHxHEIGHT
// among the set's sizes.
func (k *torchImage) size(req wire.ImageRequest) (int, int, bool) {
	p := k.sp.Set.Preset
	if req.Size == "auto" || req.Size == "" {
		return p.Width, p.Height, true
	}
	w, h, ok := strings.Cut(req.Size, "x")
	if !ok {
		return 0, 0, false
	}
	width, errW := strconv.Atoi(w)
	height, errH := strconv.Atoi(h)
	for _, s := range k.sp.Set.Sizes {
		if errW == nil && errH == nil && s == [2]int{width, height} {
			return width, height, true
		}
	}
	return 0, 0, false
}

// Unsupported refuses, before any check of rights or content and with no
// engine effect, what the engine does not do: another model, an edit, more
// than one image, a size the set did not measure, an extension, and a prompt
// the sd path's bound refuses (no control character, U+2028 or U+2029
// reaches an engine that may write it into its log).
func (k *torchImage) Unsupported(req wire.ImageRequest) string {
	switch {
	case req.Model != k.model:
		return "model:" + req.Model
	case req.Mode != wire.ImageModeGenerate || req.ImageDigest != "" || req.MaskDigest != "":
		return "mode:edit"
	case req.Count != 1:
		return "count"
	}
	if _, _, ok := k.size(req); !ok {
		return "size"
	}
	if len(req.RequiredExtensions) > 0 {
		return "extension:" + req.RequiredExtensions[0]
	}
	if req.Prompt == "" {
		return "prompt"
	}
	if err := sdcpp.CheckPrompt(req.Prompt); err != nil {
		return reasonOf(err)
	}
	return ""
}

// torchImageShape is an image's measures the dispatch row's known-bad
// values are checked against: the transformer's image tokens, one per
// 16x16 pixels, beside the 512 text tokens the pipeline pads to.
func torchImageShape(width, height int) map[string]int64 {
	return map[string]int64{"math-attention-tokens": int64(width/16)*int64(height/16) + 512}
}

// fingerprint is what an image dispatches on the GPU (PYTORCH-T8).
func (k *torchImage) fingerprint(width, height int) gpufault.Fingerprint {
	return k.shared.fingerprint(k.model, k.sp, fmt.Sprintf("%dx%d steps%d image", width, height, k.sp.Set.Preset.Steps))
}

func imageReply(outcome wire.ReplyOutcome, reason string) inference.ImageEngineReply {
	return inference.ImageEngineReply{Outcome: outcome, Reason: reason}
}

// Run makes one image: GPU admission, the engine loaded under the holder,
// the job followed to its end with progress per step and decode, and the
// engine's pixels written as the provider's own PNG (PYTORCH-T11).
func (k *torchImage) Run(ctx context.Context, model string, req wire.ImageRequest, _ map[string]inference.ImageInput, progress func(int64, string)) inference.ImageEngineReply {
	if reason := k.Unsupported(req); reason != "" || model != k.model {
		if reason == "" {
			reason = "model:" + model
		}
		return imageReply(wire.ReplyOutcomeUnsupportedFeature, reason)
	}
	emit := func(percent int64, message string) {
		if progress != nil {
			progress(percent, message)
		}
	}
	width, height, _ := k.size(req)
	k.mu.Lock()
	k.waiting++
	k.mu.Unlock()
	defer func() {
		k.mu.Lock()
		k.waiting--
		k.mu.Unlock()
	}()
	fp := k.fingerprint(width, height)
	if row, ok := gpufault.DispatchFor(gpufault.EnginePyTorchROCm); ok {
		if err := row.Check(torchImageShape(width, height)); err != nil {
			return imageReply(wire.ReplyOutcomeUnavailable, gpufault.WordKnownBad)
		}
	}
	if gpu := k.host.options.GPU; gpu != nil {
		if err := gpu.Quarantine(fp); err != nil {
			return imageReply(wire.ReplyOutcomeUnavailable, gpufault.WordQuarantined)
		}
		if word := admitImageGPU(ctx, gpu, fp, emit); word != "" {
			if ctx.Err() != nil {
				return imageReply(wire.ReplyOutcomeCancelled, "caller")
			}
			return imageReply(wire.ReplyOutcomeUnavailable, word)
		}
		// The wake lease is held from here until the job ends (RES-G5).
		defer gpu.Busy("image " + k.model)()
	}
	emit(0, "waiting for engine")
	// The set's use first, then its load, then the operation slot: the
	// lock order of PYTORCH-T10.
	end, err := k.host.resources.Begin(ctx, k.model)
	if err != nil {
		if ctx.Err() != nil {
			return imageReply(wire.ReplyOutcomeCancelled, "caller")
		}
		return imageReply(wire.ReplyOutcomeUnavailable, imageEngineLost)
	}
	defer end()
	emit(1, "loading model")
	running, release, err := k.shared.acquire(ctx, k.sp.Set.Name)
	if ctx.Err() != nil {
		if release != nil {
			release()
		}
		return imageReply(wire.ReplyOutcomeCancelled, "caller")
	}
	if err != nil {
		return imageReply(wire.ReplyOutcomeUnavailable, loadFailure(err))
	}
	defer release()
	return k.generate(ctx, running, req.Prompt, width, height, fp, emit)
}

// admitImageGPU waits while the machine is in Modern Standby or the driver
// is coming back from a reset, and returns the word that refuses the image
// otherwise, or "" once it may run (RES-G3..RES-G5, PYTORCH-G1).
func admitImageGPU(ctx context.Context, gpu *gpufault.Guard, fp gpufault.Fingerprint, emit func(int64, string)) string {
	for {
		err := gpu.AdmitRequest(ctx, fp)
		var refused *gpufault.Refusal
		if !errors.As(err, &refused) {
			return ""
		}
		if refused.Word != gpufault.WordStandby && refused.Word != gpufault.WordResetting {
			return refused.Word
		}
		emit(0, "waiting for the GPU")
		timer := time.NewTimer(gpuWait)
		select {
		case <-timer.C:
		case <-ctx.Done():
			timer.Stop()
			return refused.Word
		}
	}
}

// imageSeed is a seed drawn for one image: image@1 names none.
func imageSeed() (int64, error) {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return 0, err
	}
	return int64(binary.LittleEndian.Uint64(b[:]) >> 11), nil
}

// generate sends one image job and follows it: progress per step and
// decode, the one frame, a cancel between steps, and the engine's exit as a
// lost engine.
func (k *torchImage) generate(ctx context.Context, running *torchProcess, prompt string, width, height int,
	fp gpufault.Fingerprint, emit func(int64, string)) inference.ImageEngineReply {
	seed, err := imageSeed()
	if err != nil {
		return imageReply(wire.ReplyOutcomeUnavailable, imageEngineFailed)
	}
	id, err := operationID()
	if err != nil {
		return imageReply(wire.ReplyOutcomeUnavailable, imageEngineFailed)
	}
	name := k.sp.Set.Name
	sink := running.conn.begin(id, int64(width*height*3))
	defer running.conn.end(sink)
	job := map[string]any{"op": "generate", "set": name, "id": id, "prompt": prompt, "seed": seed, "width": width, "height": height}
	if err := running.conn.send(job, nil); err != nil {
		k.shared.stop(running, name, "image engine lost")
		return imageReply(wire.ReplyOutcomeUnavailable, k.lost(ctx, fp))
	}
	emit(5, "generating")
	share := int64(k.sp.Set.StepShare[""])
	if share <= 0 || share >= 100 {
		share = 50
	}
	deadline := time.NewTimer(k.timeLimit)
	defer deadline.Stop()
	var pixels []byte
	var cancelTimer <-chan time.Time
	callerDone := ctx.Done()
	for {
		select {
		case <-callerDone:
			callerDone = nil
			if err := running.conn.send(map[string]any{"op": "cancel", "id": id}, nil); err != nil {
				k.shared.stop(running, name, "image cancel")
				return imageReply(wire.ReplyOutcomeCancelled, "caller")
			}
			within := k.cancelWithin
			if within <= 0 {
				within = torchImageCancelWithin
			}
			timer := time.NewTimer(within)
			defer timer.Stop()
			cancelTimer = timer.C
		case <-cancelTimer:
			// The engine did not stop between steps in time; only stopping
			// it is sure to end the job.
			k.shared.stop(running, name, "image cancel")
			return imageReply(wire.ReplyOutcomeCancelled, "caller")
		case <-deadline.C:
			k.shared.stop(running, name, "image time limit")
			return imageReply(wire.ReplyOutcomeUnavailable, imageEngineTimeout)
		case <-running.conn.closed:
			k.shared.stop(running, name, "image engine lost")
			if ctx.Err() != nil {
				return imageReply(wire.ReplyOutcomeCancelled, "caller")
			}
			return imageReply(wire.ReplyOutcomeUnavailable, k.lost(ctx, fp))
		case m := <-sink.messages:
			switch m.Type {
			case "progress":
				percent := share * int64(m.Done) / int64(m.Total)
				if m.Phase == "decode" {
					percent = share + (100-share)*int64(m.Done)/int64(m.Total)
				}
				emit(max(5, min(percent, 99)), m.Phase)
			case "frame":
				if m.Index != 0 || pixels != nil {
					k.shared.stop(running, name, "image engine broke its protocol")
					return imageReply(wire.ReplyOutcomeUnavailable, imageEngineResult)
				}
				pixels = m.payload
			case "done":
				if ctx.Err() != nil {
					return imageReply(wire.ReplyOutcomeCancelled, "caller")
				}
				if m.Frames != 1 || pixels == nil {
					return imageReply(wire.ReplyOutcomeUnavailable, imageEngineResult)
				}
				k.shared.reportPeak(k.sp, m.Peak)
				data, err := rgbpng.Encode(width, height, pixels)
				if err != nil {
					return imageReply(wire.ReplyOutcomeUnavailable, imageEngineResult)
				}
				emit(100, "done")
				return inference.ImageEngineReply{Outcome: wire.ReplyOutcomeCompleted,
					Images: []inference.ImageOutput{{Data: data, MediaType: rgbpng.PNGMediaType}}}
			case "cancelled":
				if ctx.Err() == nil {
					return imageReply(wire.ReplyOutcomeUnavailable, imageEngineFailed)
				}
				return imageReply(wire.ReplyOutcomeCancelled, "caller")
			case "failed":
				if ctx.Err() != nil {
					return imageReply(wire.ReplyOutcomeCancelled, "caller")
				}
				switch m.Code {
				case "device":
					// The engine exits after a lost device; its line on the
					// log reaches the Guard, and the image reads gpu:reset
					// when a reset is recorded (RES-G1..G3).
					select {
					case <-running.process.exited:
					case <-time.After(deviceLostConfirm):
					}
					k.shared.stop(running, name, "image engine lost its device")
					return imageReply(wire.ReplyOutcomeUnavailable, k.lost(ctx, fp))
				case "input":
					return imageReply(wire.ReplyOutcomeUnavailable, imageEngineRefused)
				default:
					return imageReply(wire.ReplyOutcomeUnavailable, imageEngineFailed)
				}
			}
		}
	}
}

// lost is the word an image whose engine went away fails with: gpu:reset
// when a driver reset is recorded within the Guard's window of the loss,
// which also quarantines the request, else engine:lost (RES-G2, RES-G3).
func (k *torchImage) lost(ctx context.Context, fp gpufault.Fingerprint) string {
	gpu := k.host.options.GPU
	if gpu == nil {
		return imageEngineLost
	}
	if _, reset := gpu.Classify(ctx, time.Now()); !reset {
		return imageEngineLost
	}
	gpu.Failed(fp)
	return gpufault.WordReset
}
