package torch

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/png"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	"github.com/openabstractions/abstraction-inference/go/providerhost/sdcpp"
	"github.com/openabstractions/abstraction-inference/go/providerhost/videoenc"
	res "github.com/openabstractions/abstraction-resource/go/client"
	"github.com/openabstractions/abstraction-resource/go/gpufault"
	router "github.com/openabstractions/abstraction-router/go"
)

const torchImageModel = "flux2-klein-4b"

// Test leases: the video set's own (the fixture's 3 GiB), the image set's
// own, and the two together beside each other.
const (
	testImageLease    = 2 << 30
	testTogetherBF16  = 4 << 30
	testTogetherF32   = 5 << 30
	testImageSide     = 64
	testImageMaxBytes = testImageSide * testImageSide * 3
)

// imageSet is a FLUX.2 [klein] set of small files placed for f: the real
// set's roles and names, its tokenizer in one folder, a 64x64 size and test
// leases beside the fixture's video set.
func (f *torchFixture) imageSet(t *testing.T) TorchSetPlace {
	t.Helper()
	set := Flux2Klein4BTorch
	set.Files = nil
	set.Preset.Width, set.Preset.Height = testImageSide, testImageSide
	set.Sizes = [][2]int{{testImageSide, testImageSide}}
	set.Lease = testImageLease
	set.Beside = []TorchBeside{{Set: HummingbirdXTI2VTorch.Name, Option: "decode-dtype",
		Lease: map[string]int64{"bf16": testTogetherBF16, "f32": testTogetherF32}, Commit: 8 << 30}}
	downloads := filepath.Join(f.downloads, "image-downloads")
	for _, file := range Flux2Klein4BTorch.Files {
		data := []byte("weights of " + file.Role)
		target := filepath.Join(downloads, file.Name)
		var locators []string
		if strings.HasPrefix(file.Role, "qwen3-") {
			target = filepath.Join(downloads, "tokenizer", file.Name)
			locators = []string{"tokenizer/" + file.Name}
		}
		write(t, target, data)
		sum := sha256.Sum256(data)
		set.Files = append(set.Files, VideoModelFile{Name: file.Name, Role: file.Role, Size: int64(len(data)),
			SHA256: hex.EncodeToString(sum[:]), License: "test", Locators: locators})
	}
	state := filepath.Dir(f.engine.State)
	places := TorchModelPlaces(state, set)
	if _, err := PlaceTorchModels(context.Background(), set, []ModelSearch{FolderSearch(downloads)}, places); err != nil {
		t.Fatal(err)
	}
	return TorchSetPlace{Set: set, Places: places, ModelRecord: TorchModelRecord(state, set)}
}

// imageHost is a host serving the image set alone on the PyTorch engine.
func (f *torchFixture) imageHost(t *testing.T, leases *fakeLeases, adjust func(*Options)) *Host {
	t.Helper()
	sp := f.imageSet(t)
	engine := f.engine
	engine.Set, engine.Options, engine.Places, engine.ModelRecord = sp.Set, nil, sp.Places, sp.ModelRecord
	engine.Source = ""
	options := Options{Leases: leases, Idle: time.Hour, Renew: 50 * time.Millisecond, Wait: time.Second,
		Image: &ImageOptions{Torch: &engine}}
	if adjust != nil {
		adjust(&options)
	}
	h, err := New(options)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(h.Close)
	return h
}

// bothHost is a host whose video set and image set share one engine
// process.
func (f *torchFixture) bothHost(t *testing.T, leases *fakeLeases, adjust func(*Options)) *Host {
	t.Helper()
	sp := f.imageSet(t)
	return f.host(t, leases, func(o *Options) {
		o.Video.Torch.Beside = &sp
		o.Image = &ImageOptions{}
		if adjust != nil {
			adjust(o)
		}
	})
}

func imageRequest(prompt string) wire.ImageRequest {
	return wire.ImageRequest{Model: torchImageModel, Mode: wire.ImageModeGenerate, Prompt: prompt, Size: "auto", Count: 1}
}

// progressLog keeps an image's progress reports.
type progressLog struct {
	mu       sync.Mutex
	percents []int64
}

func (p *progressLog) add(percent int64, _ string) {
	p.mu.Lock()
	p.percents = append(p.percents, percent)
	p.mu.Unlock()
}

// wireReply is an image engine's reply as the tests read it.
type wireReply struct {
	outcome wire.ReplyOutcome
	reason  string
	images  []inference.ImageOutput
}

func runImage(t *testing.T, h *Host, ctx context.Context, request wire.ImageRequest) (wireReply, *progressLog) {
	t.Helper()
	engine, route := h.ImageEngine()
	if engine == nil || route == nil || !route.ServesModel(router.ProfileImage, torchImageModel) {
		t.Fatal("the host serves no image engine for the image set")
	}
	progress := &progressLog{}
	reply := engine.Run(ctx, torchImageModel, request, nil, progress.add)
	return wireReply{reply.Outcome, reply.Reason, reply.Images}, progress
}

// PYTORCH-T11: an image loads the engine under the image set's lease,
// sends the set's held files, its key table and preset and no source, and
// returns the provider's own PNG of the engine's pixels.
func TestTorchImageMakesAPNGThroughTheEngine(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t)
	leases := newFakeLeases()
	h := f.imageHost(t, leases, nil)
	reply, progress := runImage(t, h, context.Background(), imageRequest("a lighthouse at dusk"))
	if reply.outcome != wire.ReplyOutcomeCompleted || len(reply.images) != 1 {
		t.Fatalf("reply %+v; log %s", reply, f.log())
	}
	out := reply.images[0]
	if out.MediaType != "image/png" || http.DetectContentType(out.Data) != "image/png" {
		t.Fatalf("media %q", out.MediaType)
	}
	img, err := png.Decode(bytes.NewReader(out.Data))
	if err != nil || img.Bounds() != image.Rect(0, 0, testImageSide, testImageSide) {
		t.Fatalf("decoded %v %v", img.Bounds(), err)
	}
	// The stand-in's pixel at x, y, channel c is (7x + 3y + c) mod 256.
	for _, p := range [][2]int{{0, 0}, {5, 9}, {63, 63}} {
		r, g, b, _ := img.At(p[0], p[1]).RGBA()
		want := (7*p[0] + 3*p[1]) % 256
		if int(r>>8) != want || int(g>>8) != (want+1)%256 || int(b>>8) != (want+2)%256 {
			t.Fatalf("pixel %v: %d %d %d", p, r>>8, g>>8, b>>8)
		}
	}
	if len(leases.grants) != 1 || leases.grants[0] != testImageLease {
		t.Fatalf("grants %v", leases.grants)
	}
	if !slices.IsSorted(progress.percents) || len(progress.percents) < 4 || progress.percents[len(progress.percents)-1] != 100 {
		t.Fatalf("progress %v", progress.percents)
	}
	records := f.records(t)
	if len(records) != 2 || records[0]["event"] != "load" || records[1]["event"] != "generate" || records[0]["set"] != torchImageModel {
		t.Fatalf("records %v", records)
	}
	load := records[0]["load"].(map[string]any)
	if src := load["source"].(map[string]any); len(src) != 0 {
		t.Fatalf("an image set was sent source %v", src)
	}
	tables := load["tables"].(map[string]any)
	if len(tables) != 1 || tables["flux-vae"] == nil {
		t.Fatalf("tables %v", tables)
	}
	if preset := load["preset"].(map[string]any); preset["steps"] != float64(4) || preset["guidance"] != float64(1) || len(preset) != 2 {
		t.Fatalf("preset %v", preset)
	}
	if files := load["files"].(map[string]any); len(files) != len(Flux2Klein4BTorch.Files) {
		t.Fatalf("files %v", files)
	}
	job := records[1]
	if job["set"] != torchImageModel || job["width"] != float64(testImageSide) || job["frames"] != float64(1) {
		t.Fatalf("job %v", job)
	}
	// KIT-P3: the session logs the image job's start and the set's
	// phases as it does a video job's, and never the prompt. The provider
	// copies the engine's stderr on its own goroutine, so each line is
	// awaited.
	for _, line := range []string{`"event": "generate", "id": "` + job["id"].(string) + `", "set": "` + torchImageModel + `"`,
		`"event": "text_encode", "id": "` + job["id"].(string) + `", "at": "start"`,
		`"event": "text_encode", "id": "` + job["id"].(string) + `", "at": "end"`,
		`"event": "load", "set": "` + torchImageModel + `"`} {
		if !f.awaitLog(line, 10*time.Second) {
			t.Fatalf("no %s in the engine log:\n%s", line, f.log())
		}
	}
	if log := f.log(); strings.Contains(log, "lighthouse") {
		t.Fatalf("the prompt reached the engine log:\n%s", log)
	}
}

// PYTORCH-T11: what the engine does not do is refused before any lease
// or engine: another model, an edit, two images, an unmeasured size, an
// extension, an empty prompt and the sd path's prompt bound.
func TestTorchImageRefusesWhatTheEngineDoesNotDo(t *testing.T) {
	f := newTorchFixture(t)
	leases := newFakeLeases()
	h := f.imageHost(t, leases, nil)
	for want, change := range map[string]func(*wire.ImageRequest){
		"model:other":           func(r *wire.ImageRequest) { r.Model = "other" },
		"mode:edit":             func(r *wire.ImageRequest) { r.Mode = wire.ImageModeEdit },
		"count":                 func(r *wire.ImageRequest) { r.Count = 2 },
		"size":                  func(r *wire.ImageRequest) { r.Size = "1024x1024" },
		"extension:modelhost/x": func(r *wire.ImageRequest) { r.RequiredExtensions = []string{"modelhost/x"} },
		"prompt":                func(r *wire.ImageRequest) { r.Prompt = "" },
		sdcpp.ReasonVideoBound:  func(r *wire.ImageRequest) { r.Prompt = "line\nbreak" },
	} {
		request := imageRequest("a lighthouse")
		change(&request)
		if got := h.image.Unsupported(request); got != want {
			t.Fatalf("%s: %q", want, got)
		}
		if reply, _ := runImage(t, h, context.Background(), request); reply.outcome != wire.ReplyOutcomeUnsupportedFeature {
			t.Fatalf("%s: ran %+v", want, reply)
		}
	}
	if acquires, _, _ := leases.counts(); acquires != 0 || len(f.records(t)) != 0 {
		t.Fatalf("an engine started for a refused request: %d", acquires)
	}
}

// PYTORCH-T10: the video set and the image set share one process. The
// set that starts it holds its own lease, bound to the process; the set
// that joins holds the two sets' lease together less the first's, bound to
// nothing. Unloading the joined set unloads it in the engine and gives its
// lease back while the process serves on; unloading the first stops the
// process and both leases go back.
func TestTorchVideoAndImageShareOneProcess(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t)
	leases := newFakeLeases()
	h := f.bothHost(t, leases, nil)
	if o := awaitVideo(t, h, startVideo(t, h, f.request(t, "job-1")), wire.VideoStateCompleted, wire.VideoStateFailed); o.Status != wire.VideoStateCompleted {
		t.Fatalf("video %+v; log %s", o, f.log())
	}
	h.torch.mu.Lock()
	first := h.torch.running
	h.torch.mu.Unlock()
	if reply, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse")); reply.outcome != wire.ReplyOutcomeCompleted {
		t.Fatalf("image %+v; log %s", reply, f.log())
	}
	h.torch.mu.Lock()
	same := h.torch.running == first && h.torch.base == torchTestModel
	h.torch.mu.Unlock()
	// The video set's option is AMD's F32 decode: its row together.
	if !same || !slices.Equal(leases.grants, []int64{3 << 30, testTogetherF32 - 3<<30}) {
		t.Fatalf("one process %v, grants %v", same, leases.grants)
	}
	videoLease, imageLease := h.Lease(torchTestModel), h.Lease(torchImageModel)
	leases.mu.Lock()
	videoBound, imageBound := leases.bindings[videoLease], leases.bindings[imageLease]
	leases.mu.Unlock()
	if videoBound != first.process.PID() || imageBound != 0 {
		t.Fatalf("bindings: video %d image %d, process %d", videoBound, imageBound, first.process.PID())
	}
	// A second image reuses the loaded set: no lease, no load.
	if reply, _ := runImage(t, h, context.Background(), imageRequest("another lighthouse")); reply.outcome != wire.ReplyOutcomeCompleted || len(leases.grants) != 2 {
		t.Fatalf("second image %+v grants %v", reply, leases.grants)
	}

	h.Unload(torchImageModel, "test")
	if _, released, _ := leases.counts(); len(released) != 1 || released[0] != imageLease || h.Loaded(torchImageModel) {
		t.Fatalf("released %v", released)
	}
	if first.exited() || !first.conn.loaded(torchTestModel) || first.conn.loaded(torchImageModel) {
		t.Fatalf("after the image unload: exited %v video %v image %v", first.exited(), first.conn.loaded(torchTestModel), first.conn.loaded(torchImageModel))
	}
	if o := awaitVideo(t, h, startVideo(t, h, f.request(t, "job-2")), wire.VideoStateCompleted, wire.VideoStateFailed); o.Status != wire.VideoStateCompleted {
		t.Fatalf("video after the image unload %+v", o)
	}
	if reply, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse again")); reply.outcome != wire.ReplyOutcomeCompleted {
		t.Fatalf("image again %+v", reply)
	}
	if len(leases.grants) != 3 || leases.grants[2] != testTogetherF32-3<<30 {
		t.Fatalf("grants %v", leases.grants)
	}

	// The holder's own check, every renewal, gives back the lease of a set
	// whose process is gone.
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go h.Run(ctx)
	h.Unload(torchTestModel, "test")
	select {
	case <-first.process.exited:
	case <-time.After(30 * time.Second):
		t.Fatal("the process survived its own set's unload")
	}
	waitUntil(t, func() bool { _, released, _ := leases.counts(); return len(released) == 3 })
	if h.Loaded(torchImageModel) || h.Loaded(torchTestModel) {
		t.Fatal("a set stayed resident after its process stopped")
	}
	events := []string{}
	for _, r := range f.records(t) {
		if r["event"] == "load" || r["event"] == "unload" {
			events = append(events, r["event"].(string)+" "+r["set"].(string))
		}
	}
	if !slices.Equal(events, []string{"load " + torchTestModel, "load " + torchImageModel, "unload " + torchImageModel, "load " + torchImageModel}) {
		t.Fatalf("engine events %v", events)
	}
}

// PYTORCH-T10: either set may start the process; the video set joins an
// image engine with the lease together less the image set's, by the video
// set's decode option.
func TestTorchVideoJoinsAnImageEngine(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t)
	leases := newFakeLeases()
	h := f.bothHost(t, leases, func(o *Options) { o.Video.Torch.Options = map[string]string{"decode-dtype": "bf16"} })
	if reply, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse")); reply.outcome != wire.ReplyOutcomeCompleted {
		t.Fatalf("image %+v; log %s", reply, f.log())
	}
	if o := awaitVideo(t, h, startVideo(t, h, f.request(t, "job-1")), wire.VideoStateCompleted, wire.VideoStateFailed); o.Status != wire.VideoStateCompleted {
		t.Fatalf("video %+v; log %s", o, f.log())
	}
	if !slices.Equal(leases.grants, []int64{testImageLease, testTogetherBF16 - testImageLease}) {
		t.Fatalf("grants %v", leases.grants)
	}
	h.torch.mu.Lock()
	base := h.torch.base
	h.torch.mu.Unlock()
	if base != torchImageModel {
		t.Fatalf("base %q", base)
	}
}

// KIT-P3: when the image set starts the process the video set shares,
// the exit record names the image set, the one the process was started for.
func TestTorchExitRecordNamesTheSetThatStartedTheProcess(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t)
	exits := &engineExits{}
	h := f.bothHost(t, newFakeLeases(), func(o *Options) { o.OnEngineExit = exits.hook })
	if reply, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse")); reply.outcome != wire.ReplyOutcomeCompleted {
		t.Fatalf("image %+v; log %s", reply, f.log())
	}
	h.Close()
	if x := exits.await(t); x.Engine != "oa-torch-engine" || x.Model != torchImageModel || !x.Stopped {
		t.Fatalf("exit record %+v", x)
	}
}

// PYTORCH-T10: the set whose lease the process is bound to is in use
// while the other set works, so neither the idle clock nor a yield stops
// the process under a running job.
func TestTorchFirstSetStaysWhileTheJoinedSetWorks(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t, "--step=0.3", "--steps=5")
	leases := newFakeLeases()
	h := f.bothHost(t, leases, nil)
	if reply, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse")); reply.outcome != wire.ReplyOutcomeCompleted {
		t.Fatalf("image %+v", reply)
	}
	op := startVideo(t, h, f.request(t, "job-1"))
	waitUntil(t, func() bool { return h.Loaded(torchTestModel) })
	imageLease := h.Lease(torchImageModel)
	if !h.torch.busy(torchImageModel) {
		t.Fatal("the process's own set is idle while the joined set works")
	}
	answer, reason := h.yield(res.YieldRequest{Resource: DefaultResource, Lease: imageLease, Amount: testImageLease})
	if answer != res.YieldAnswerRefused || !strings.Contains(reason, "active") {
		t.Fatalf("yield %s %q", answer, reason)
	}
	if o := awaitVideo(t, h, op, wire.VideoStateCompleted, wire.VideoStateFailed); o.Status != wire.VideoStateCompleted {
		t.Fatalf("video %+v", o)
	}
	if h.torch.busy(torchImageModel) {
		t.Fatal("the process's own set stays busy after the work ended")
	}
}

// PYTORCH-T10, review of f6d3e3fbd: a joined set's idle time runs out
// while the other set's job runs, and the set stays, because every set is
// busy while an engine operation holds the process: the holder keeps
// restarting its idle time. The video completes, and the image set goes
// after it. Each job takes sixteen steps of 0.5 s, 8 s: the image's idle
// time of 11 s, counted from its load, ends 3 s into the second video,
// which leaves 3 s for the gap between the two jobs under a loaded machine
// and 5 s before the video ends.
func TestTorchJoinedSetIsNotIdledUnderARunningJob(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t, "--step=0.5", "--steps=16")
	leases := newFakeLeases()
	h := f.bothHost(t, leases, func(o *Options) { o.Idle = 11 * time.Second })
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go h.Run(ctx)
	if o := awaitVideo(t, h, startVideo(t, h, f.request(t, "job-1")), wire.VideoStateCompleted, wire.VideoStateFailed); o.Status != wire.VideoStateCompleted {
		t.Fatalf("first video %+v; log %s", o, f.log())
	}
	if reply, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse")); reply.outcome != wire.ReplyOutcomeCompleted {
		t.Fatalf("image %+v", reply)
	}
	o, stayed, idle := followSecondVideo(t, h, f)
	if o.Status != wire.VideoStateCompleted {
		t.Fatalf("second video %+v; log %s", o, f.log())
	}
	if !stayed || idle {
		t.Fatalf("under the running video: image loaded throughout %v, image ever idle %v", stayed, idle)
	}
	waitUntil(t, func() bool { return !h.Loaded(torchImageModel) })
}

// followSecondVideo runs a second video and watches the image set while the
// video's job holds the process: whether it stayed loaded in the engine and
// whether it ever read idle to the holder.
func followSecondVideo(t *testing.T, h *Host, f *torchFixture) (wire.VideoObservation, bool, bool) {
	t.Helper()
	op := startVideo(t, h, f.request(t, "job-2"))
	done := make(chan wire.VideoObservation, 1)
	go func() { done <- awaitVideo(t, h, op, wire.VideoStateCompleted, wire.VideoStateFailed) }()
	stayed, idle := true, false
	for {
		select {
		case o := <-done:
			return o, stayed, idle
		case <-time.After(50 * time.Millisecond):
			h.torch.mu.Lock()
			running := h.torch.running
			h.torch.mu.Unlock()
			if running == nil || len(h.torch.slot) == 0 {
				continue
			}
			if !running.conn.loaded(torchImageModel) {
				stayed = false
			}
			if !h.torch.busy(torchImageModel) {
				idle = true
			}
		}
	}
}

// PYTORCH-T10, review of f6d3e3fbd: an unload of the joined set asked
// while the other set's job runs is an engine operation: it waits for the
// job, which completes, and then unloads the set in the engine.
func TestTorchUnloadWaitsForTheRunningJob(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t, "--step=0.3", "--steps=8")
	leases := newFakeLeases()
	h := f.bothHost(t, leases, nil)
	// The engine answers an unload within a second here; the second video
	// takes 2.4 s, so an unload sent while it runs would stop the process.
	h.torch.unloadWithin = time.Second
	if o := awaitVideo(t, h, startVideo(t, h, f.request(t, "job-1")), wire.VideoStateCompleted, wire.VideoStateFailed); o.Status != wire.VideoStateCompleted {
		t.Fatalf("first video %+v; log %s", o, f.log())
	}
	if reply, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse")); reply.outcome != wire.ReplyOutcomeCompleted {
		t.Fatalf("image %+v", reply)
	}
	op := startVideo(t, h, f.request(t, "job-2"))
	waitUntil(t, func() bool { return len(h.torch.slot) > 0 && h.video.active() })
	h.Unload(torchImageModel, "test")
	if o := awaitVideo(t, h, op, wire.VideoStateCompleted, wire.VideoStateFailed); o.Status != wire.VideoStateCompleted {
		t.Fatalf("second video %+v; log %s", o, f.log())
	}
	events := []string{}
	for _, r := range f.records(t) {
		if r["event"] == "generate" || r["event"] == "unload" {
			events = append(events, r["event"].(string)+" "+r["set"].(string))
		}
	}
	want := []string{"generate " + torchTestModel, "generate " + torchImageModel, "generate " + torchTestModel, "unload " + torchImageModel}
	if !slices.Equal(events, want) || h.Loaded(torchImageModel) || !h.Loaded(torchTestModel) {
		t.Fatalf("engine events %v, image %v, video %v", events, h.Loaded(torchImageModel), h.Loaded(torchTestModel))
	}
}

// PYTORCH-T10, delta review of 46682621b: the lock order is a set's
// holder component lock, then the operation slot. An unload that holds the
// image set's component lock waits for the slot, which something else
// holds; an image job arriving meanwhile takes the set's use first and so
// waits for that lock, not the slot. Once the slot is free both finish.
func TestTorchUnloadAndJobKeepTheLockOrder(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t)
	leases := newFakeLeases()
	h := f.bothHost(t, leases, nil)
	if o := awaitVideo(t, h, startVideo(t, h, f.request(t, "job-1")), wire.VideoStateCompleted, wire.VideoStateFailed); o.Status != wire.VideoStateCompleted {
		t.Fatalf("video %+v; log %s", o, f.log())
	}
	if reply, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse")); reply.outcome != wire.ReplyOutcomeCompleted {
		t.Fatalf("image %+v", reply)
	}
	// Something holds the slot between a busy check and a Stop.
	h.torch.slot <- struct{}{}
	unloaded := make(chan struct{})
	go func() {
		h.Unload(torchImageModel, "test")
		close(unloaded)
	}()
	time.Sleep(200 * time.Millisecond)
	replied := make(chan wireReply, 1)
	go func() {
		reply, _ := runImage(t, h, context.Background(), imageRequest("another lighthouse"))
		replied <- reply
	}()
	time.Sleep(200 * time.Millisecond)
	<-h.torch.slot
	select {
	case <-unloaded:
	case <-time.After(30 * time.Second):
		t.Fatal("the unload never finished: a lock-order deadlock")
	}
	select {
	case reply := <-replied:
		if reply.outcome != wire.ReplyOutcomeCompleted {
			t.Fatalf("image %+v", reply)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("the image job never finished: a lock-order deadlock")
	}
}

// PYTORCH-T10, RES-G2: a driver reset stops the PyTorch engine at once,
// without waiting for the running job or the operation slot. The image set
// joined the video set's process, so the holder stops it first (by name)
// while the video's job holds the slot; each job takes 6 s, and the reset
// returns within 2.
func TestTorchResetStopsTheEngineUnderARunningJob(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t, "--step=0.3", "--steps=20")
	leases := newFakeLeases()
	h := f.bothHost(t, leases, nil)
	if o := awaitVideo(t, h, startVideo(t, h, f.request(t, "job-1")), wire.VideoStateCompleted, wire.VideoStateFailed); o.Status != wire.VideoStateCompleted {
		t.Fatalf("video %+v; log %s", o, f.log())
	}
	if reply, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse")); reply.outcome != wire.ReplyOutcomeCompleted {
		t.Fatalf("image %+v", reply)
	}
	op := startVideo(t, h, f.request(t, "job-2"))
	waitUntil(t, func() bool { return len(h.torch.slot) > 0 && h.video.active() })
	h.torch.mu.Lock()
	running := h.torch.running
	h.torch.mu.Unlock()
	start := time.Now()
	lost := torchLoser{h}.LoseAll("test reset")
	if took := time.Since(start); took > 2*time.Second || !running.exited() {
		t.Fatalf("the reset took %s, process gone %v", took, running.exited())
	}
	// The video's own job sees its engine gone and may give the video set
	// back before the holder reaches it; either way both sets end released.
	for _, l := range lost {
		if !l.Released {
			t.Fatalf("lost %+v", lost)
		}
	}
	if len(lost) == 0 || lost[0].Name != torchImageModel {
		t.Fatalf("lost %+v", lost)
	}
	if o := awaitVideo(t, h, op, wire.VideoStateCompleted, wire.VideoStateFailed); o.Status != wire.VideoStateFailed {
		t.Fatalf("the video under the reset %+v", o)
	}
	waitUntil(t, func() bool {
		_, released, _ := leases.counts()
		return len(released) == 2 && !h.Loaded(torchTestModel) && !h.Loaded(torchImageModel)
	})
}

// PYTORCH-T10, delta review of 46682621b: a join whose job is cancelled
// while the engine is still loading the set leaves no set loaded without a
// lease: its Stop sends the unload inside the join's own operation, and the
// lease goes back only once the engine answered unloaded.
func TestTorchCancelledJoinUnloadsBeforeTheLeaseGoes(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t, "--image-mode=slow-load", "--load-seconds=2")
	leases := newFakeLeases()
	h := f.bothHost(t, leases, nil)
	if o := awaitVideo(t, h, startVideo(t, h, f.request(t, "job-1")), wire.VideoStateCompleted, wire.VideoStateFailed); o.Status != wire.VideoStateCompleted {
		t.Fatalf("video %+v; log %s", o, f.log())
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		for deadline := time.Now().Add(30 * time.Second); time.Now().Before(deadline); time.Sleep(20 * time.Millisecond) {
			h.torch.mu.Lock()
			running := h.torch.last
			h.torch.mu.Unlock()
			if running != nil && running.conn.state(torchImageModel) != nil {
				break
			}
		}
		time.Sleep(300 * time.Millisecond)
		cancel()
	}()
	if reply, _ := runImage(t, h, ctx, imageRequest("a lighthouse")); reply.outcome != wire.ReplyOutcomeCancelled {
		t.Fatalf("image %+v", reply)
	}
	acquires, released, _ := leases.counts()
	if acquires != 2 || len(released) != 1 {
		t.Fatalf("acquires %d released %v", acquires, released)
	}
	// The release came after the engine's unloaded, which follows its
	// record of the unload.
	unload := false
	for _, r := range f.records(t) {
		unload = unload || r["event"] == "unload" && r["set"] == torchImageModel
	}
	if !unload || h.Loaded(torchImageModel) {
		t.Fatalf("unload recorded %v, loaded %v", unload, h.Loaded(torchImageModel))
	}
	if !h.Loaded(torchTestModel) {
		t.Fatal("the video set left with the cancelled join")
	}
}

// PYTORCH-T7: options the encoder refuses and a platform with no system
// encoder read encoder:unavailable; any other encode failure, such as a
// file error, reads encoder:failed.
func TestEncodeFailureReasons(t *testing.T) {
	for err, want := range map[error]string{
		fmt.Errorf("x: %w", videoenc.ErrOptions):                                videoEncoderUnavailable,
		videoenc.ErrNoSystemEncoder:                                             videoEncoderUnavailable,
		&os.PathError{Op: "write", Path: "v.mp4", Err: errors.New("disk full")}: videoEncoderFailed,
	} {
		if got := encodeFailure(err); got != want {
			t.Fatalf("%v: %q, want %q", err, got, want)
		}
	}
}

// PYTORCH-T10: a start that proves files for minutes holds only its own
// lock; busy, which the holder reads while it renews every lease, answers
// at once.
func TestTorchBusyAnswersDuringAStart(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t)
	h := f.imageHost(t, newFakeLeases(), nil)
	entered, release := make(chan struct{}), make(chan struct{})
	h.torch.beforeStart = func() {
		close(entered)
		<-release
	}
	result := make(chan wireReply, 1)
	go func() {
		reply, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse"))
		result <- reply
	}()
	select {
	case <-entered:
	case <-time.After(30 * time.Second):
		t.Fatal("the start never began")
	}
	answered := make(chan bool, 1)
	go func() { answered <- h.torch.busy(torchImageModel) }()
	select {
	case busy := <-answered:
		if !busy {
			t.Fatal("the set reads idle while its load holds the process")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("busy blocked behind a start")
	}
	close(release)
	if reply := <-result; reply.outcome != wire.ReplyOutcomeCompleted {
		t.Fatalf("image %+v", reply)
	}
}

// PYTORCH-T10: a set that fails to join leaves the running set serving
// and gives its own lease back.
func TestTorchFailedJoinKeepsTheRunningSet(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t, "--image-mode=error-load")
	leases := newFakeLeases()
	reported := make(chan error, 8)
	h := f.bothHost(t, leases, func(o *Options) { o.OnError = func(err error) { reported <- err } })
	if o := awaitVideo(t, h, startVideo(t, h, f.request(t, "job-1")), wire.VideoStateCompleted, wire.VideoStateFailed); o.Status != wire.VideoStateCompleted {
		t.Fatalf("video %+v", o)
	}
	reply, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse"))
	if reply.outcome != wire.ReplyOutcomeUnavailable || reply.reason != NoEngine {
		t.Fatalf("image %+v", reply)
	}
	if _, released, _ := leases.counts(); len(released) != 1 || h.Loaded(torchImageModel) || !h.Loaded(torchTestModel) {
		t.Fatalf("released %v image %v video %v", released, h.Loaded(torchImageModel), h.Loaded(torchTestModel))
	}
	found := false
	for len(reported) > 0 {
		err := <-reported
		found = found || strings.Contains(err.Error(), "code files")
	}
	if !found {
		t.Fatal("the refused join's code was not reported")
	}
	if o := awaitVideo(t, h, startVideo(t, h, f.request(t, "job-2")), wire.VideoStateCompleted, wire.VideoStateFailed); o.Status != wire.VideoStateCompleted {
		t.Fatalf("video after the failed join %+v", o)
	}
}

// PYTORCH-T11: a cancel between steps ends the image cancelled and keeps
// the engine.
func TestTorchImageCancelKeepsTheEngine(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t, "--step=0.3", "--steps=8")
	leases := newFakeLeases()
	h := f.imageHost(t, leases, nil)
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		for deadline := time.Now().Add(30 * time.Second); !h.Loaded(torchImageModel) && time.Now().Before(deadline); {
			time.Sleep(20 * time.Millisecond)
		}
		time.Sleep(400 * time.Millisecond)
		cancel()
	}()
	reply, _ := runImage(t, h, ctx, imageRequest("a lighthouse"))
	if reply.outcome != wire.ReplyOutcomeCancelled {
		t.Fatalf("reply %+v", reply)
	}
	if !h.Loaded(torchImageModel) {
		t.Fatal("the cancel stopped the engine")
	}
	waitUntil(t, func() bool {
		for _, r := range f.records(t) {
			if r["event"] == "after_cancel" {
				return true
			}
		}
		return false
	})
}

// PYTORCH-T8, T11: a lost device during an image is a reset when the
// engine's exit confirms it: the image fails gpu:reset, its fingerprint is
// quarantined, and the identical request is refused before any lease.
func TestTorchImageDeviceLostIsAReset(t *testing.T) {
	engineRuns(t)
	f := newTorchFixture(t, "--image-mode=device-lost")
	r := newGPURig(t)
	leases := newFakeLeases()
	h := f.imageHost(t, leases, func(o *Options) { o.GPU = r.guard })
	reply, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse"))
	if reply.outcome != wire.ReplyOutcomeUnavailable || reply.reason != gpufault.WordReset {
		t.Fatalf("reply %+v; log %s", reply, f.log())
	}
	fault := nextGPUFault(t, r.faults)
	if len(fault.Quarantined) != 1 || !strings.Contains(fault.Quarantined[0].Fingerprint.Model, "set=torch:"+torchImageModel) {
		t.Fatalf("fault %+v", fault)
	}
	acquires, _, _ := leases.counts()
	if again, _ := runImage(t, h, context.Background(), imageRequest("a lighthouse")); again.outcome != wire.ReplyOutcomeUnavailable || again.reason != gpufault.WordQuarantined {
		t.Fatalf("the identical request ran: %+v", again)
	}
	if after, _, _ := leases.counts(); after != acquires {
		t.Fatal("a quarantined image asked for a lease")
	}
}

// PYTORCH-T2, T10: the FLUX.2 [klein] row is enforceable as written, its
// figures beside Hummingbird-XT read from either set, and its VAE key table
// names every Diffusers tensor once from a distinct Comfy-Org tensor.
func TestFluxTorchRowIsValid(t *testing.T) {
	s := Flux2Klein4BTorch
	if err := s.valid(); err != nil {
		t.Fatal(err)
	}
	if s.Profile != torchProfileImage || s.Lease != gibTenths(185) || s.Preset.Steps != 4 || s.FirstFrame != FirstFrameNone {
		t.Fatalf("row %+v", s)
	}
	// The live walkthrough of 2026-10-02 (D9) reserved 18,828,230,656 bytes
	// for one 512x512 image; the lease covers that peak with about 5 percent
	// above it.
	const walkthroughPeak = 18_828_230_656
	if s.Lease < walkthroughPeak*105/100 {
		t.Fatalf("lease %d is below the measured peak %d with 5 percent above it", s.Lease, int64(walkthroughPeak))
	}
	video := HummingbirdXTI2VTorch
	for _, c := range []struct {
		options map[string]string
		want    int64
	}{{nil, gibTenths(444)}, {map[string]string{"decode-dtype": "bf16"}, gibTenths(409)}} {
		a, commit, ok := s.beside(video, c.options, nil)
		b, _, okB := video.beside(s, nil, c.options)
		if !ok || !okB || a != c.want || b != c.want || commit != 55<<30 {
			t.Fatalf("beside %v: %d %d %v %v", c.options, a, b, ok, okB)
		}
	}
	// The joined set's lease is the two together less the first set's own.
	both := newTorchShared(nil, TorchEngine{Set: video, Beside: &TorchSetPlace{Set: s}})
	for _, c := range []struct {
		join, base string
		want       int64
	}{
		{s.Name, video.Name, gibTenths(444) - gibTenths(310)},
		{video.Name, s.Name, gibTenths(444) - gibTenths(185)},
	} {
		sp, _ := both.engine.place(c.join)
		if got := both.leaseFor(sp, c.base); got != c.want {
			t.Fatalf("%s joining %s: %d, want %d", c.join, c.base, got, c.want)
		}
	}
	var keys struct {
		Paired int               `json:"paired"`
		Table  map[string]string `json:"table"`
	}
	if err := json.Unmarshal(torchFluxVAEKeys, &keys); err != nil {
		t.Fatal(err)
	}
	sources := map[string]bool{}
	for to, from := range keys.Table {
		if to == "" || from == "" || sources[from] {
			t.Fatalf("table entry %q <- %q", to, from)
		}
		sources[from] = true
	}
	if keys.Paired != 250 || len(keys.Table) != 251 || keys.Table["decoder.mid_block.attentions.0.to_q.weight"] != "decoder.mid.attn_1.q.weight" {
		t.Fatalf("table of %d (%d paired)", len(keys.Table), keys.Paired)
	}
	broken := s
	broken.Beside = []TorchBeside{{Set: video.Name, Option: "decode-dtype", Lease: map[string]int64{"bf16": s.Lease}, Commit: 1}}
	if broken.valid() == nil {
		t.Fatal("a lease together no larger than the set's own passed")
	}
	if _, err := os.Stat(filepath.Join("torch", "flux2-vae-keys.json")); err != nil {
		t.Fatal(err)
	}
}
