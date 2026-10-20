package torch

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"time"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	"github.com/openabstractions/abstraction-inference/go/providerhost/sdcpp"
	"github.com/openabstractions/abstraction-inference/go/providerhost/videoenc"
	"github.com/openabstractions/abstraction-resource/go/gpufault"
)

// torchCancelWithin bounds how long a running video's cancel waits for the
// engine to stop between two steps or decode chunks before its process is
// stopped. A step at 512x512x81 took 5.4 to 7.2 s and a decode chunk about
// 3 s (research/live-runs/2026-10-01-rocm.md).
const torchCancelWithin = 60 * time.Second

// torchKind runs a video on the OA PyTorch engine (PYTORCH-T1..T10): the
// engine process it shares with an image set when one is configured.
type torchKind struct {
	v      *videoHost
	shared *torchShared
	sp     TorchSetPlace

	// cancelWithin replaces torchCancelWithin; only tests set it.
	cancelWithin time.Duration
	// softwareOnly asks videoenc for Microsoft's software encoder alone;
	// only tests set it, as the hardware encoder runs only under
	// supervision there.
	softwareOnly bool
}

// The reasons a completed PyTorch video carries: which H.264 encoder wrote
// its MP4 (PYTORCH-T7). A failed one reads encoder:unavailable when no
// system encoder can write it (options it refuses, or no system encoder on
// the platform), and encoder:failed when the encode itself failed, a disk
// or file error included, which the provider reports with its own words.
const (
	videoEncoderHardware    = "encoder:hardware"
	videoEncoderSoftware    = "encoder:software"
	videoEncoderUnavailable = "encoder:unavailable"
	videoEncoderFailed      = "encoder:failed"
)

func (k *torchKind) Bounds() videoBounds { return k.sp.Set.bounds() }

// shape is the request's measures the dispatch row's known-bad values are
// checked against: the clip's own token count, latent frames times the
// 32-pixel patch grid (the seq_len the engine sets).
func torchShape(p sdcpp.VideoParams) map[string]int64 {
	tokens := int64((p.Frames-1)/4+1) * int64(p.Height/32) * int64(p.Width/32)
	return map[string]int64{"math-attention-tokens": tokens}
}

// admit refuses a prompt the sd path's bound refuses, and a request whose
// shape reaches a value the pytorch-rocm dispatch row records as hanging the
// GPU (resource RES-G5).
func (k *torchKind) Admit(params sdcpp.VideoParams) (wire.StartOutcome, string) {
	// The sd path's own prompt bound: no control character, U+2028 or U+2029
	// reaches an engine that may write the prompt into its log.
	if err := sdcpp.CheckPrompt(params.Prompt); err != nil {
		return wire.StartOutcomeUnsupportedFeature, reasonOf(err)
	}
	if row, ok := gpufault.DispatchFor(gpufault.EnginePyTorchROCm); ok {
		if err := row.Check(torchShape(params)); err != nil {
			return wire.StartOutcomeUnavailable, gpufault.WordKnownBad
		}
	}
	if params.Width%32 != 0 || params.Height%32 != 0 || (params.Frames-1)%4 != 0 {
		return wire.StartOutcomeUnsupportedFeature, sdcpp.ReasonVideoBound
	}
	return wire.StartOutcomeAccepted, ""
}

// fingerprint is the model set with its options, the engine (its script's
// digest and the reviewed stack) and the request's shape; the prompt and
// seed are not part of it (PYTORCH-T8).
func (k *torchKind) Fingerprint(p sdcpp.VideoParams) gpufault.Fingerprint {
	return k.shared.fingerprint(k.v.opts.Model, k.sp,
		fmt.Sprintf("%dx%dx%d fps%d image=%t", p.Width, p.Height, p.Frames, p.FPS, len(p.InitImage) > 0))
}

// fingerprint is what a request for sp dispatches on the GPU: the model set
// with its options, the engine (its script's digest and the reviewed stack)
// and shape; the prompt and seed are not part of it (PYTORCH-T8).
func (s *torchShared) fingerprint(model string, sp TorchSetPlace, shape string) gpufault.Fingerprint {
	var options []string
	for _, o := range sp.Set.Options {
		options = append(options, o.Name+"="+sp.Set.option(sp.Options, o.Name))
	}
	script := s.engine.script
	if script == nil {
		script = torchEngineScript
	}
	sum := sha256.Sum256(script)
	return gpufault.Fingerprint{Model: model + " set=torch:" + sp.Set.Name + " options=" + strings.Join(options, " "),
		Engine: "oa-torch-engine " + hex.EncodeToString(sum[:6]) + " " + s.engine.runtime.Release,
		Shape:  shape}
}

// rgb decodes the provider's own first-frame PNG to the engine's raw RGB.
func rgb(frame []byte, width, height int) ([]byte, error) {
	img, err := png.Decode(bytes.NewReader(frame))
	if err != nil {
		return nil, err
	}
	if img.Bounds() != image.Rect(0, 0, width, height) {
		return nil, fmt.Errorf("the first frame is %v, not %dx%d", img.Bounds(), width, height)
	}
	out := make([]byte, 0, width*height*3)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			out = append(out, byte(r>>8), byte(g>>8), byte(b>>8))
		}
	}
	return out, nil
}

// run loads the engine, sends the job and follows it: progress per step and
// per decode chunk, the frames into the provider's own MP4, a cancel between
// steps, and the engine's exit as a lost engine (PYTORCH-T6, T7).
func (k *torchKind) Run(ctx context.Context, op *videoOp) {
	v := k.v
	// videoHost.run holds the set's use (holder Begin) for the whole run;
	// acquire loads the set and then holds the operation slot.
	running, release, err := k.shared.acquire(ctx, k.sp.Set.Name)
	if cancelled(op) {
		if release != nil {
			release()
		}
		return
	}
	if err != nil {
		v.Finish(op, wire.VideoStateFailed, loadFailure(err), nil)
		return
	}
	defer release()
	params := v.Params(op)
	frame, err := rgb(params.InitImage, params.Width, params.Height)
	if err != nil {
		v.Finish(op, wire.VideoStateFailed, videoEngineRefused, nil)
		return
	}
	frameBytes := int64(params.Width * params.Height * 3)
	sink := running.conn.begin(op.ID(), frameBytes)
	defer running.conn.end(sink)
	job := map[string]any{"op": "generate", "set": k.sp.Set.Name, "id": op.ID(), "prompt": params.Prompt, "seed": params.Seed,
		"width": params.Width, "height": params.Height, "frames": params.Frames, "fps": params.FPS}
	if err := running.conn.send(job, frame); err != nil {
		k.shared.stop(running, k.sp.Set.Name, "video engine lost")
		v.Finish(op, wire.VideoStateFailed, v.Lost(ctx, op), nil)
		return
	}
	v.SetState(op, wire.VideoStateGenerating, 0)
	share := k.sp.Set.StepShare[k.sp.Set.option(k.sp.Options, "decode-dtype")]
	if share <= 0 || share >= 100 {
		share = 50
	}
	deadline := time.NewTimer(v.TimeLimit())
	defer deadline.Stop()
	frames := make([][]byte, 0, params.Frames)
	var cancelTimer <-chan time.Time
	cancelSent := false
	cancelCh := op.Cancel()
	for {
		select {
		case <-cancelCh:
			if !cancelSent {
				cancelSent = true
				if err := running.conn.send(map[string]any{"op": "cancel", "id": op.ID()}, nil); err != nil {
					k.shared.stop(running, k.sp.Set.Name, "video cancel")
					return
				}
				within := k.cancelWithin
				if within <= 0 {
					within = torchCancelWithin
				}
				timer := time.NewTimer(within)
				defer timer.Stop()
				cancelTimer = timer.C
			}
			cancelCh = nil
			continue
		case <-cancelTimer:
			// The engine did not stop between steps in time; only stopping
			// it is sure to end the job.
			k.shared.stop(running, k.sp.Set.Name, "video cancel")
			return
		case <-v.Done():
			v.Finish(op, wire.VideoStateFailed, videoEngineLost, nil)
			return
		case <-deadline.C:
			k.shared.stop(running, k.sp.Set.Name, "video time limit")
			v.Finish(op, wire.VideoStateFailed, videoEngineTimeout, nil)
			return
		case <-running.conn.closed:
			k.shared.stop(running, k.sp.Set.Name, "video engine lost")
			if !cancelSent {
				v.Finish(op, wire.VideoStateFailed, v.Lost(ctx, op), nil)
			}
			return
		case m := <-sink.messages:
			switch m.Type {
			case "progress":
				percent := share * m.Done / m.Total
				if m.Phase == "decode" {
					percent = share + (100-share)*m.Done/m.Total
				}
				v.SetPercent(op, percent)
			case "frame":
				if m.Index != len(frames) || len(frames) >= params.Frames {
					k.shared.stop(running, k.sp.Set.Name, "video engine broke its protocol")
					v.Finish(op, wire.VideoStateFailed, videoEngineResult, nil)
					return
				}
				frames = append(frames, m.payload)
			case "done":
				if cancelSent {
					// The job ended before the cancel reached it; the
					// operation is already cancelled.
					return
				}
				k.complete(op, params, frames, m)
				return
			case "cancelled":
				if !cancelSent {
					v.Finish(op, wire.VideoStateFailed, videoEngineFailed, nil)
				}
				return
			case "failed":
				if cancelSent {
					return
				}
				switch m.Code {
				case "device":
					// The engine exits after a lost device; its line on the
					// log reaches the Guard, and the video reads gpu:reset
					// when a reset is recorded (RES-G1..G3).
					select {
					case <-running.process.exited:
					case <-time.After(deviceLostConfirm):
					}
					k.shared.stop(running, k.sp.Set.Name, "video engine lost its device")
					v.Finish(op, wire.VideoStateFailed, v.Lost(ctx, op), nil)
				case "input":
					v.Finish(op, wire.VideoStateFailed, videoEngineRefused, nil)
				default:
					v.Finish(op, wire.VideoStateFailed, videoEngineFailed, nil)
				}
				return
			}
		}
	}
}

// complete checks the job's end against the request, encodes the frames as
// the provider's own H.264 MP4 through the provider kit (INF-VE1) and keeps
// it (INF-G3); the completed video's reason names the encoder that ran. A
// peak above the leases of the process's resident sets is reported:
// residency above the estimate stays charged (RES-L2).
func (k *torchKind) complete(op *videoOp, params sdcpp.VideoParams, frames [][]byte, m torchMessage) {
	v := k.v
	if m.Frames != params.Frames || len(frames) != params.Frames {
		v.Finish(op, wire.VideoStateFailed, videoEngineResult, nil)
		return
	}
	k.shared.reportPeak(k.sp, m.Peak)
	data, encoded, err := k.encode(params, frames)
	if err != nil {
		reason := encodeFailure(err)
		if v.host.options.OnError != nil {
			v.host.options.OnError(fmt.Errorf("pytorch: encoding the %s video: %w", k.sp.Set.Name, err))
		}
		v.Finish(op, wire.VideoStateFailed, reason, nil)
		return
	}
	sum := sha256.Sum256(data)
	digest := "sha256:" + hex.EncodeToString(sum[:])
	if err := v.Keep(digest, data); err != nil {
		v.Finish(op, wire.VideoStateFailed, "content:unavailable", nil)
		return
	}
	reason := videoEncoderSoftware
	if encoded.Hardware {
		reason = videoEncoderHardware
	}
	v.Finish(op, wire.VideoStateCompleted, reason, &wire.Delivery{Digest: digest, MediaType: videoenc.MP4MediaType, Size: int64(len(data)), Location: "local"})
}

// encodeFailure is the reason a video whose encode failed with err ends
// with: options the encoder refuses and a platform with no system encoder
// mean no encoder can write it; any other failure, such as a disk or file
// error, is the encode's own, which the provider reports as it is.
func encodeFailure(err error) string {
	if errors.Is(err, videoenc.ErrOptions) || errors.Is(err, videoenc.ErrNoSystemEncoder) {
		return videoEncoderUnavailable
	}
	return videoEncoderFailed
}

// encode writes frames as one MP4 through a private folder in the engine's
// state: the hardware H.264 encoder when the machine has one, and
// Microsoft's software encoder when there is none or the hardware encoder
// fails.
func (k *torchKind) encode(params sdcpp.VideoParams, frames [][]byte) ([]byte, videoenc.Encoded, error) {
	dir := filepath.Join(k.shared.engine.State, "encode")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, videoenc.Encoded{}, err
	}
	options := videoenc.Options{Width: params.Width, Height: params.Height, FPS: params.FPS, Layout: videoenc.RGB24,
		Software: k.softwareOnly}
	data, encoded, err := videoenc.Encode(dir, options, frames)
	if err == nil || options.Software || errors.Is(err, videoenc.ErrOptions) || errors.Is(err, videoenc.ErrNoSystemEncoder) {
		return data, encoded, err
	}
	if k.v.host.options.OnError != nil {
		k.v.host.options.OnError(fmt.Errorf("pytorch: the hardware H.264 encoder failed (%v); encoding in software", err))
	}
	options.Software = true
	return videoenc.Encode(dir, options, frames)
}
