package torch

import (
	"context"

	"github.com/openabstractions/abstraction-identity/go/trustedpath"
	"github.com/openabstractions/abstraction-provider-kit/go/engineproc"
	"github.com/openabstractions/abstraction-provider-kit/go/modelplace"
	"github.com/openabstractions/abstraction-provider-kit/go/refusal"
	"github.com/openabstractions/abstraction-provider-kit/go/videoops"
)

// The PyTorch provider's names for the provider kit's parts it shares with
// every local-engine provider (abstraction-provider-kit CONTRACT.md): the
// engine process and its exit record, the typed refusal, the placed model
// files and the video@1 operation table.

// EngineExit is one engine's end (KIT-P3).
type EngineExit = engineproc.Exit

type ownedSpec = engineproc.Spec

type engineLog = engineproc.Log

// engineLogBytes bounds each engine log file; one earlier file is kept.
const engineLogBytes = engineproc.LogBytes

func openEngineLog(path string, max int64) (*engineLog, error) { return engineproc.OpenLog(path, max) }

const (
	stopWithin        = engineproc.StopWithin
	deviceLostConfirm = engineproc.DeviceLostConfirm
)

// Refusal is a typed refusal: a word a caller reads and a detail the log
// keeps (KIT-R1).
type Refusal = refusal.Refusal

func refuse(word, format string, args ...any) *Refusal { return refusal.New(word, format, args...) }

// WordOf is err's refusal word, or "".
func WordOf(err error) string { return refusal.WordOf(err) }

// loadFailure is the word a video or image whose engine could not load
// fails with.
func loadFailure(err error) string { return refusal.LoadWord(err) }

// gpuRefusal is a GPU fault refusal as this provider's own refusal word
// (resource RES-G3).
func gpuRefusal(err error) error { return refusal.GPU(err) }

// VideoModelFile is one pinned weight file of a model set (KIT-M1).
type VideoModelFile = modelplace.File

// UsePolicy is a publisher's policy of permitted use beside a licence.
type UsePolicy = modelplace.UsePolicy

// ModelSearch, FolderSearch and StoreSearch name where a pinned file may
// lie (KIT-M1).
type (
	ModelSearch  = modelplace.Search
	FolderSearch = modelplace.FolderSearch
	StoreSearch  = modelplace.StoreSearch
)

func lowerHex(s string, n int) bool { return modelplace.LowerHex(s, n) }

func formatOwners(o trustedpath.Owners) string { return modelplace.FormatOwners(o) }

func parseOwners(s string) (trustedpath.Owners, bool) { return modelplace.ParseOwners(s) }

func validFiles(set string, files []VideoModelFile) error { return modelplace.ValidFiles(set, files) }

func placeFiles(ctx context.Context, files []VideoModelFile, searches []ModelSearch, places string) (map[string]string, error) {
	return modelplace.Place(ctx, files, searches, places)
}

// readFilePlaces reads a places record; a refusal names the models install
// that writes it.
func readFilePlaces(places, set string, files []VideoModelFile) (map[string]string, map[string]trustedpath.Owners, error) {
	return modelplace.Read(places, set, files, "pytorchd models install --set "+set)
}

// ModelsRoot is the folder OA keeps its own model sets in under home.
func ModelsRoot(home string) string { return modelplace.ModelsRoot(home) }

// AdoptModelFolder makes dir a model folder of this account when it is a set
// folder directly under root (KIT-M3).
func AdoptModelFolder(root, dir string, linkRoots []string) (bool, error) {
	return modelplace.AdoptModelFolder(root, dir, linkRoots)
}

// SetFolderOf is the folder directly under root that holds p, or "".
func SetFolderOf(root, p string) string { return modelplace.SetFolderOf(root, p) }

// FirstFrameUse says how a video model set takes a caller's first frame.
type FirstFrameUse = videoops.FirstFrameUse

const (
	FirstFrameOptional = videoops.FirstFrameOptional
	FirstFrameRequired = videoops.FirstFrameRequired
	FirstFrameNone     = videoops.FirstFrameNone
)

type (
	videoOp     = videoops.Op
	videoBounds = videoops.Bounds
)

const (
	DefaultVideoTimeLimit = videoops.DefaultTimeLimit
	DefaultVideoReady     = videoops.DefaultReady
	MaxVideoQueue         = videoops.MaxQueue
	MaxVideoOperations    = videoops.MaxOperations
	gpuWait               = videoops.GPUWait
)

// Video reasons an Admission or an Observe carries.
const (
	videoQueueFull     = videoops.ReasonQueueFull
	videoTableFull     = videoops.ReasonTableFull
	videoEngineFailed  = videoops.ReasonEngineFailed
	videoEngineResult  = videoops.ReasonEngineResult
	videoEngineTimeout = videoops.ReasonEngineTimeout
	videoEngineLost    = videoops.ReasonEngineLost
	videoEngineRefused = videoops.ReasonEngineRefused
	videoFrameRequired = videoops.ReasonFrameRequired
)

func reasonOf(err error) string { return videoops.ReasonOf(err) }

func operationID() (string, error) { return videoops.OperationID() }

func cancelled(op *videoOp) bool { return op.Cancelled() }
