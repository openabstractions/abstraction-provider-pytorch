package torch

import (
	"errors"
	"fmt"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"github.com/openabstractions/abstraction-inference/go/providerhost/sdcpp"
)

// torchReview is the security review of the PyTorch-on-ROCm stack the OA
// PyTorch engine runs on (CONTRACT.md PYTORCH-T1).
const torchReview = "research/security-reviews/pytorch-rocm-windows-2.11.0-rocm7.13.0.md"

// Roles of a Hummingbird-XT PyTorch set's files, as the engine script names
// them in its load message.
const (
	torchRoleTransformer      = "transformer"
	torchRoleTurboVAED        = "turbovaed"
	torchRoleWanVAE           = "wan-vae"
	torchRoleTokenizer        = "tokenizer"
	torchRoleTokenizerConfig  = "tokenizer-config"
	torchRoleTokenizerSpecial = "tokenizer-special"
	torchRoleTokenizerSpiece  = "tokenizer-spiece"
)

// The inference profiles a PyTorch model set serves.
const (
	torchProfileVideo = "video"
	torchProfileImage = "image"
)

// A TorchSourceFile is one file of AMD's reviewed Hummingbird-XT source the
// engine runs, by the module name the engine imports it as and its path
// below the source folder, pinned by size and SHA-256 (its Git blob at the
// reviewed commit was checked when it was fetched).
type TorchSourceFile struct {
	Module string
	Path   string
	Size   int64
	SHA256 string
}

// A TorchPreset is the generation preset the provider sends the engine:
// for a video set AMD's timestep table, flow shift, guidance and MoE
// boundary; for an image set the step count and guidance; and the rate and
// size a request of rate zero and size auto gets.
type TorchPreset struct {
	Timesteps []int
	Shift     float64
	Guidance  float64
	Boundary  float64
	// Steps is an image set's denoising step count.
	Steps  int
	FPS    int
	Width  int
	Height int
}

// A TorchBeside is what two model sets resident in one engine process
// measured together: the other set, the option of that set the figures
// depend on, the device memory reserved by that option's value, and the
// process's peak commit with a margin (PYTORCH-T10).
type TorchBeside struct {
	Set    string
	Option string
	Lease  map[string]int64
	Commit uint64
}

// A TorchOption is one model-set option a person may choose with
// --video-model-option NAME=VALUE: its allowed values, the first the
// default.
type TorchOption struct {
	Name   string
	Values []string
}

// A TorchModelSet is one model set the OA PyTorch engine serves as data: the
// name callers ask for, its files by role, the source the engine runs, the
// preset and bounds, and the lease and commit limit measured for it
// (CONTRACT.md PYTORCH-T2, T5).
type TorchModelSet struct {
	Name   string
	Review string
	// Profile is the inference profile the set serves: video or image.
	Profile string
	Files   []VideoModelFile
	Source  []TorchSourceFile
	Preset  TorchPreset
	// Sizes, MaxFrames and MaxFPS bound a caller's request.
	Sizes     [][2]int
	MaxFrames int
	MaxFPS    int
	// FirstFrame is how the set takes a caller's first frame.
	FirstFrame FirstFrameUse
	UsePolicy  UsePolicy
	// Lease is the device memory the set holds while resident, in bytes:
	// the measured peak reserved with every option the set allows.
	Lease int64
	// Commit is the engine job's per-process memory limit on Windows, in
	// bytes: the measured peak commit with a margin.
	Commit uint64
	// Options are the choices a person may make for the set.
	Options []TorchOption
	// StepShare is the percent of a video's progress the denoising steps
	// fill per decode precision; the decode fills the rest. An image set's
	// share is keyed "".
	StepShare map[string]int
	// Beside are the measured figures of this set and another resident in
	// the same engine process.
	Beside []TorchBeside
}

// amdUsePolicy is AMD's Responsible AI Use Policy, which travels with the
// Hummingbird-XT weights under the Open RAIL-MS licence.
var amdUsePolicy = UsePolicy{Name: "AMD Responsible AI Use Policy", License: "Open RAIL-MS",
	URL: "https://github.com/AMD-AGI/HummingbirdXT/blob/929e90a26c2d023c89f4428aa43869c88817a55e/NOTICES"}

// gibTenths is n tenths of a GiB, the unit the research harness measured
// (torch.cuda.max_memory_reserved / 2**30).
func gibTenths(n int64) int64 { return n * (1 << 30) / 10 }

// TorchModelSets are the reviewed PyTorch model sets, by name.
var TorchModelSets = []TorchModelSet{HummingbirdXTI2VTorch, Flux2Klein4BTorch}

// TorchModelSetNamed is the set of TorchModelSets named name.
func TorchModelSetNamed(name string) (TorchModelSet, bool) {
	for _, s := range TorchModelSets {
		if s.Name == name {
			return s, true
		}
	}
	return TorchModelSet{}, false
}

// HummingbirdXTI2VTorch is AMD's Hummingbird-XT image to video with its own
// TurboVAED decoder on PyTorch for ROCm, the model set MuseTree's Idea Clip
// runs, as research/live-runs/2026-10-01-rocm.md ran it: the i2v transformer
// and TurboVAED where ASUS's AI store keeps them, the Wan 2.2 VAE and umT5
// encoder from Wan-AI's own Diffusers safetensors, and the umT5 tokenizer
// from amd/HummingbirdXT, each by SHA-256 (research/security-reviews/
// hummingbird-xt-i2v-files.md, pytorch-rocm-windows-2.11.0-rocm7.13.0.md).
var HummingbirdXTI2VTorch = TorchModelSet{
	Name:    "hummingbird-xt-i2v",
	Review:  torchReview,
	Profile: torchProfileVideo,
	// AMD's infer/run_i2v.sh at 929e90a2: timesteps 1000, 750 and 500 at
	// shift 10, guidance 1, boundary 0.875; MuseTree's Idea Clip size.
	Preset:     TorchPreset{Timesteps: []int{1000, 750, 500}, Shift: 10, Guidance: 1, Boundary: 0.875, FPS: 16, Width: 512, Height: 512},
	Sizes:      [][2]int{{512, 512}},
	MaxFrames:  81,
	MaxFPS:     16,
	FirstFrame: FirstFrameRequired,
	UsePolicy:  amdUsePolicy,
	// The diagnostic rerun of 2026-10-02 (feedback/gpu-runs-2026-10-02.md
	// item 22) reserved 31.51 GB (31,509,708,800 bytes) at 512x512x81 with
	// AMD's F32 decode, above the 29.3 GiB run c'' measured; the lease is
	// that peak with about 5 percent above it. The BF16 decode peaked lower
	// (run B2, 24.3 GiB allocated against 26.3 to 26.5 with F32).
	Lease: gibTenths(310),
	// Run c'': 32.7 GiB peak process commit, with about 30 percent above it.
	Commit:  43 << 30,
	Options: []TorchOption{{Name: "decode-dtype", Values: []string{"f32", "bf16"}}},
	// Run c'' (F32 decode): 17.5 s for three steps, 19 s decode; run B2
	// (BF16): 16.5 s steps, 5.2 s decode.
	StepShare: map[string]int{"f32": 48, "bf16": 76},
	Files: []VideoModelFile{
		{Name: "diffusion_pytorch_model_i2v.safetensors", Role: torchRoleTransformer, Size: 9999659672,
			SHA256: "6937e10a78a130d5f3a4d44705db476d98af5932b3cad1e011180294ca2b8291",
			Source: "https://huggingface.co/amd/HummingbirdXT", Revision: "4d3bd2e3a8c96a189ed158ab74b575ec6a424d2c", License: "Apache-2.0",
			Locators: []string{"hummingbird/HM_2.2_5B/transformer_20251218/diffusion_pytorch_model_i2v.safetensors"}},
		{Name: "turbovaed-wan22_v1_tiling_16_12.safetensors", Role: torchRoleTurboVAED, Size: 152049060,
			SHA256: "04be976c369c5ce0d48cbe6685244fdb40fc42a23594d7c3c4470c5610b5d09f",
			Source: "https://huggingface.co/amd/HummingbirdXT", Revision: "4d3bd2e3a8c96a189ed158ab74b575ec6a424d2c", License: "Apache-2.0",
			Locators: []string{"hummingbird/vae_decoder/wan22_v1_tiling_16_12/diffusion_pytorch_model.safetensors",
				"vae/wan22_v1_tiling_16_12/diffusion_pytorch_model.safetensors"}},
		{Name: "wan2.2-vae-diffusers.safetensors", Role: torchRoleWanVAE, Size: 2818777808,
			SHA256: "62cd18f19438e35b32ac63020e2852f566e9b02f46b6cdbd87972a356e3c6f4b",
			Source: "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers", Revision: "b8fff7315c768468a5333511427288870b2e9635", License: "Apache-2.0",
			Locators: []string{"wan-diffusers/vae/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors"}},
		{Name: "model-00001-of-00003.safetensors", Role: "umt5-1", Size: 4935812536,
			SHA256: "a8e861969c7433e707cc5a74065d795d36cca07ec96eb6763eb4083df7248f58",
			Source: "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers", Revision: "b8fff7315c768468a5333511427288870b2e9635", License: "Apache-2.0",
			Locators: []string{"wan-diffusers/text_encoder/model-00001-of-00003.safetensors", "text_encoder/model-00001-of-00003.safetensors"}},
		{Name: "model-00002-of-00003.safetensors", Role: "umt5-2", Size: 4983103192,
			SHA256: "d57d948ece4837d850b7a859a4415121d57cacf8b9ee1d4db200c67f592902d7",
			Source: "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers", Revision: "b8fff7315c768468a5333511427288870b2e9635", License: "Apache-2.0",
			Locators: []string{"wan-diffusers/text_encoder/model-00002-of-00003.safetensors", "text_encoder/model-00002-of-00003.safetensors"}},
		{Name: "model-00003-of-00003.safetensors", Role: "umt5-3", Size: 1442935480,
			SHA256: "0da9ee284e21d1406df708788db1d502d95d75f69faa25cd26151bf8829b7c5f",
			Source: "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers", Revision: "b8fff7315c768468a5333511427288870b2e9635", License: "Apache-2.0",
			Locators: []string{"wan-diffusers/text_encoder/model-00003-of-00003.safetensors", "text_encoder/model-00003-of-00003.safetensors"}},
		// The umT5 tokenizer amd/HummingbirdXT ships beside its transformer
		// (hm_2.2_5B/google/umt5-xxl); the four files lie in one folder.
		{Name: "tokenizer.json", Role: torchRoleTokenizer, Size: 16837417,
			SHA256: "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b",
			Source: "https://huggingface.co/amd/HummingbirdXT", Revision: "4d3bd2e3a8c96a189ed158ab74b575ec6a424d2c", License: "Apache-2.0",
			Locators: []string{"umt5-xxl/tokenizer.json", "hummingbird/umt5-xxl/tokenizer.json", "hm_2.2_5B/google/umt5-xxl/tokenizer.json"}},
		{Name: "tokenizer_config.json", Role: torchRoleTokenizerConfig, Size: 61728,
			SHA256:      "ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05",
			GitBlobSHA1: "4e1cc1cd85599ce0b47fd0a746af188fe4043ff2",
			Source:      "https://huggingface.co/amd/HummingbirdXT", Revision: "4d3bd2e3a8c96a189ed158ab74b575ec6a424d2c", License: "Apache-2.0",
			Locators: []string{"umt5-xxl/tokenizer_config.json", "hummingbird/umt5-xxl/tokenizer_config.json", "hm_2.2_5B/google/umt5-xxl/tokenizer_config.json"}},
		{Name: "special_tokens_map.json", Role: torchRoleTokenizerSpecial, Size: 6623,
			SHA256:      "7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d",
			GitBlobSHA1: "14855e7052ffbb595057dfd791d293c1c940db2c",
			Source:      "https://huggingface.co/amd/HummingbirdXT", Revision: "4d3bd2e3a8c96a189ed158ab74b575ec6a424d2c", License: "Apache-2.0",
			Locators: []string{"umt5-xxl/special_tokens_map.json", "hummingbird/umt5-xxl/special_tokens_map.json", "hm_2.2_5B/google/umt5-xxl/special_tokens_map.json"}},
		{Name: "spiece.model", Role: torchRoleTokenizerSpiece, Size: 4548313,
			SHA256: "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458",
			Source: "https://huggingface.co/amd/HummingbirdXT", Revision: "4d3bd2e3a8c96a189ed158ab74b575ec6a424d2c", License: "Apache-2.0",
			Locators: []string{"umt5-xxl/spiece.model", "hummingbird/umt5-xxl/spiece.model", "hm_2.2_5B/google/umt5-xxl/spiece.model"}},
	},
	// AMD-AGI/HummingbirdXT at 929e90a26c2d023c89f4428aa43869c88817a55e,
	// infer/, fetched as GitHub's tarball of that commit with every file's Git
	// blob checked (tools/fetch_source.py; the stack review's artifact table).
	Source: []TorchSourceFile{
		{Module: "videox_fun.models.attention", Path: "videox_fun/models/attention.py", Size: 5434,
			SHA256: "ae19397021c50cd41be562cf328afeb832e5e8c28decbf2645b34cd631a43e44"},
		{Module: "videox_fun.models.wan22_turbo", Path: "videox_fun/models/wan22_turbo.py", Size: 37656,
			SHA256: "a29e4ada3add255375f34736f51a2c0f0e74604df6330b377fe69c4b92d97ef5"},
		{Module: "videox_fun.models.wan_vae3_8", Path: "videox_fun/models/wan_vae3_8.py", Size: 33508,
			SHA256: "da7f43705b3c0c89f4c3f9d4e20427a970d3865a4ccdd81ba62f508cde95c94d"},
		{Module: "videox_fun.models.wan_text_encoder", Path: "videox_fun/models/wan_text_encoder.py", Size: 15426,
			SHA256: "c01aa669ebd1fe43eb04f22f765458b641ee5d3f72e5ec7f683bef6e96b7a3c7"},
		{Module: "videox_fun.utils.fm_solvers", Path: "videox_fun/utils/fm_solvers.py", Size: 40228,
			SHA256: "f083ac101adc628c68e156a8a8cbc705020793d76d981b172dcacf33e73f43eb"},
		{Module: "videox_fun.utils.fm_solvers_unipc", Path: "videox_fun/utils/fm_solvers_unipc.py", Size: 32644,
			SHA256: "54c58e557d0f7dfaf118546aa6a8637ce1a006c88f66e9141870ebc157627935"},
		{Module: "videox_fun.pipeline.pipeline_wan2_2_ti2v", Path: "videox_fun/pipeline/pipeline_wan2_2_ti2v.py", Size: 40870,
			SHA256: "7085aad68b6b6c6d56b5376431796e66b322a201dd1e585240802733c332d3ba"},
		{Module: "autoencoder_kl_turbo_vaed_ours_wan22", Path: "examples/wan2.2/autoencoder_kl_turbo_vaed_ours_wan22.py", Size: 61943,
			SHA256: "cfce4db251f02645db70bd067b54378025b139b1e3a6357fa42ef48b2ecdb9ca"},
	},
}

// Roles of a FLUX.2 [klein] set's files, as the engine script names them in
// its load message.
const (
	torchRoleFluxTransformer = "flux-transformer"
	torchRoleFluxVAE         = "flux-vae"
	torchRoleQwen3           = "qwen3"
)

// fluxKlein names the files of black-forest-labs/FLUX.2-klein-4B at the
// revision research/live-runs/2026-10-01-rocm.md fetched its tokenizer from.
const (
	fluxKleinSource   = "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B"
	fluxKleinRevision = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
	comfyKleinSource  = "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b"
	comfyKleinRev     = "5f526678002e43af5551dadb73ce2e8c91b43afe"
)

// fluxTokenizerFile is one file of FLUX.2 [klein]'s Qwen3 tokenizer folder,
// pinned by size, SHA-256 and the Git blob SHA-1 it was fetched against.
func fluxTokenizerFile(name, role string, size int64, sha256, blob string) VideoModelFile {
	return VideoModelFile{Name: name, Role: role, Size: size, SHA256: sha256, GitBlobSHA1: blob,
		Source: fluxKleinSource, Revision: fluxKleinRevision, License: "Apache-2.0",
		Locators: []string{"flux2-klein-4b/tokenizer/" + name, "tokenizer/" + name}}
}

// Flux2Klein4BTorch is Black Forest Labs' FLUX.2 [klein] 4B text to image on
// PyTorch for ROCm, the start image of MuseTree's Idea Clip, as
// research/live-runs/2026-10-02.md ran it: 512x512 in four steps with
// guidance 1 (the distilled model needs no classifier-free guidance). Its
// weights are Comfy-Org's single files in OA's model folders, the
// transformer and VAE under Black Forest Labs' own tensor names and the
// Qwen3 4B text encoder under Hugging Face's; its tokenizer is BFL's.
// Every tensor the engine builds from them equals, byte for byte, a tensor
// of the files the live run loaded: the transformer through Diffusers'
// FLUX.2 key conversion, the VAE through flux2-vae-keys.json once rounded
// to BF16, and the text encoder by name except the MLP of its last layer,
// which the pipeline never reads (it takes layers 9, 18 and 27).
var Flux2Klein4BTorch = TorchModelSet{
	Name:    "flux2-klein-4b",
	Review:  torchReview,
	Profile: torchProfileImage,
	// Diffusers' Flux2KleinPipeline as the live run called it.
	Preset:     TorchPreset{Steps: 4, Guidance: 1, Width: 512, Height: 512},
	Sizes:      [][2]int{{512, 512}},
	FirstFrame: FirstFrameNone,
	// Phase 1 of 2026-10-02: 16.4 GiB allocated, 17.5 GiB reserved with
	// FLUX.2 loaded and the start image made. The live walkthrough of the
	// same day (research/live-runs/2026-10-02-walkthrough.md, scene 2, D9)
	// reserved 18,828,230,656 bytes (17.54 GiB) for one 512x512 image,
	// above a 17.5 GiB lease; the lease is that peak with about 5 percent
	// above it, as for Hummingbird-XT.
	Lease: gibTenths(185),
	// Phase 1: 19.0 GB peak process commit, with about 30 percent above it.
	Commit:    25 << 30,
	StepShare: map[string]int{"": 85},
	// Both sets in one process, as MuseTree's engine holds them: phase 1 of
	// 2026-10-02 reserved 40.9 GiB with the BF16 decode (run E of
	// 2026-10-01: 40.5) and 44.4 GiB with the F32 decode; the process
	// committed 42.6 and 48.3 GB at those peaks.
	Beside: []TorchBeside{{Set: "hummingbird-xt-i2v", Option: "decode-dtype",
		Lease: map[string]int64{"bf16": gibTenths(409), "f32": gibTenths(444)}, Commit: 55 << 30}},
	Files: []VideoModelFile{
		{Name: "flux-2-klein-4b.safetensors", Role: torchRoleFluxTransformer, Size: 7751105712,
			SHA256: "ec3d4e733a771f61c052fb4856c48b336c55eaf2c65487c2a1faeb9bbda7a343",
			Source: comfyKleinSource, Revision: comfyKleinRev, License: "Apache-2.0",
			Locators: []string{"flux2-klein/flux-2-klein-4b.safetensors", "diffusion_models/flux-2-klein-4b.safetensors",
				"split_files/diffusion_models/flux-2-klein-4b.safetensors"}},
		{Name: "flux2-vae.safetensors", Role: torchRoleFluxVAE, Size: 336211292,
			SHA256: "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3",
			Source: comfyKleinSource, Revision: comfyKleinRev, License: "Apache-2.0",
			Locators: []string{"flux2-klein/flux2-vae.safetensors", "vae/flux2-vae.safetensors", "split_files/vae/flux2-vae.safetensors"}},
		{Name: "qwen_3_4b.safetensors", Role: torchRoleQwen3, Size: 8044982048,
			SHA256: "6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a",
			Source: comfyKleinSource, Revision: comfyKleinRev, License: "Apache-2.0",
			Locators: []string{"zimage/qwen_3_4b.safetensors", "text_encoders/qwen_3_4b.safetensors",
				"split_files/text_encoders/qwen_3_4b.safetensors"}},
		// BFL's Qwen3 tokenizer folder; the seven files lie in one folder.
		fluxTokenizerFile("tokenizer.json", "qwen3-tokenizer", 11422654,
			"aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4", ""),
		fluxTokenizerFile("tokenizer_config.json", "qwen3-tokenizer-config", 5404,
			"443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0", "ddaf69808214a44fdd26d3785b66c1367c78277a"),
		fluxTokenizerFile("special_tokens_map.json", "qwen3-tokenizer-special", 613,
			"76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd", "ac23c0aaa2434523c494330aeb79c58395378103"),
		fluxTokenizerFile("added_tokens.json", "qwen3-tokenizer-added", 707,
			"c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680", "b54f9135e44c1e81047e8d05cb027af8bc039eed"),
		fluxTokenizerFile("chat_template.jinja", "qwen3-chat-template", 4168,
			"a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8", "01be9b307daa2d425f7c168c9fb145a286e0afb4"),
		fluxTokenizerFile("vocab.json", "qwen3-vocab", 2776833,
			"ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910", "4783fe10ac3adce15ac8f358ef5462739852c569"),
		fluxTokenizerFile("merges.txt", "qwen3-merges", 1671853,
			"8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5", "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7"),
	},
}

// beside is the measured lease and commit of s and other resident in one
// engine process, with other's options, read from either set's figures.
func (s TorchModelSet) beside(other TorchModelSet, otherOptions map[string]string, options map[string]string) (int64, uint64, bool) {
	for _, pair := range []struct {
		from, to TorchModelSet
		chosen   map[string]string
	}{{s, other, otherOptions}, {other, s, options}} {
		for _, b := range pair.from.Beside {
			if b.Set != pair.to.Name {
				continue
			}
			lease, ok := b.Lease[pair.to.option(pair.chosen, b.Option)]
			return lease, b.Commit, ok
		}
	}
	return 0, 0, false
}

// TorchCommitTogether is the per-process memory limit of an engine serving
// a, with a person's options, and b in one process: their measured commit
// together with its margin, or the larger of their own when they were never
// measured together (PYTORCH-T10).
func TorchCommitTogether(a TorchModelSet, options map[string]string, b TorchModelSet) uint64 {
	if _, commit, ok := a.beside(b, nil, options); ok {
		return commit
	}
	return max(a.Commit, b.Commit)
}

// bounds is what the set holds a caller to.
func (s TorchModelSet) bounds() videoBounds {
	return videoBounds{Width: s.Preset.Width, Height: s.Preset.Height, FPS: s.Preset.FPS, Sizes: s.Sizes,
		MaxFrames: s.MaxFrames, MaxFPS: s.MaxFPS, FirstFrame: s.FirstFrame}
}

// option is the chosen value of the named option in chosen, or its default.
func (s TorchModelSet) option(chosen map[string]string, name string) string {
	for _, o := range s.Options {
		if o.Name == name {
			if v, ok := chosen[name]; ok {
				return v
			}
			return o.Values[0]
		}
	}
	return ""
}

// checkOptions refuses an option the set does not offer or a value it does
// not allow.
func (s TorchModelSet) checkOptions(chosen map[string]string) error {
	for name, value := range chosen {
		i := slices.IndexFunc(s.Options, func(o TorchOption) bool { return o.Name == name })
		if i < 0 {
			return fmt.Errorf("the model set %s has no option %q", s.Name, name)
		}
		if !slices.Contains(s.Options[i].Values, value) {
			return fmt.Errorf("the model set %s's option %s is one of %s, not %q", s.Name, name, strings.Join(s.Options[i].Values, ", "), value)
		}
	}
	return nil
}

// ParseTorchOptions reads NAME=VALUE entries of --video-model-option and
// refuses an option set does not offer or a value it does not allow.
func ParseTorchOptions(set TorchModelSet, entries []string) (map[string]string, error) {
	out := map[string]string{}
	for _, e := range entries {
		name, value, ok := strings.Cut(e, "=")
		if !ok || name == "" || value == "" {
			return nil, fmt.Errorf("--video-model-option takes NAME=VALUE, not %q", e)
		}
		if _, seen := out[name]; seen {
			return nil, fmt.Errorf("--video-model-option names %s twice", name)
		}
		out[name] = value
	}
	return out, set.checkOptions(out)
}

// valid refuses a set that cannot be enforced as written.
func (s TorchModelSet) valid() error {
	// A video set runs AMD's reviewed source; an image set runs the venv's
	// reviewed packages alone.
	if s.Name == "" || len(s.Files) == 0 || s.Profile == torchProfileVideo && len(s.Source) == 0 {
		return errors.New("the PyTorch model set names no files or no source")
	}
	if s.Profile != torchProfileVideo && s.Profile != torchProfileImage {
		return fmt.Errorf("the PyTorch model set's profile %q is neither video nor image", s.Profile)
	}
	if err := validFiles(s.Name, s.Files); err != nil {
		return err
	}
	modules := map[string]bool{}
	for _, f := range s.Source {
		if f.Module == "" || modules[f.Module] || f.Path == "" || path.IsAbs(f.Path) || path.Clean(f.Path) != f.Path ||
			strings.HasPrefix(f.Path, "../") || strings.ContainsAny(f.Path, `\:`) || f.Size < 1 || !lowerHex(f.SHA256, 64) {
			return fmt.Errorf("the PyTorch model set's source file %q is not one pinned, clean relative path", f.Path)
		}
		modules[f.Module] = true
	}
	p := s.Preset
	switch {
	case p.Width < 32 || p.Height < 32 || p.Width%32 != 0 || p.Height%32 != 0:
		return errors.New("the PyTorch model set's preset size is not a multiple of 32")
	case !slices.Contains(s.Sizes, [2]int{p.Width, p.Height}):
		return errors.New("the PyTorch model set's preset size is not one of its sizes")
	case s.Lease < 1:
		return errors.New("the PyTorch model set has no measured lease")
	}
	if s.Profile == torchProfileImage {
		switch {
		case p.Steps < 1 || p.Steps > 50 || len(p.Timesteps) != 0:
			return errors.New("the PyTorch image set's preset is not 1 to 50 steps")
		case s.FirstFrame != FirstFrameNone || s.MaxFrames != 0 || s.MaxFPS != 0 || p.FPS != 0:
			return errors.New("the PyTorch image set names frames, a rate or a first frame")
		}
	} else {
		switch {
		case len(p.Timesteps) == 0 || len(p.Timesteps) > 50 || p.Steps != 0:
			return errors.New("the PyTorch model set's preset has no timesteps")
		case s.MaxFrames < 1 || sdcpp.AlignFrames(s.MaxFrames) != s.MaxFrames || s.MaxFrames > sdcpp.MaxVideoFrames:
			return fmt.Errorf("the PyTorch model set's %d frames is not 4n+1 within the engine's bound", s.MaxFrames)
		case s.MaxFPS < 1 || s.MaxFPS > sdcpp.MaxVideoFPS || p.FPS < 1 || p.FPS > s.MaxFPS:
			return fmt.Errorf("the PyTorch model set's rate %d is outside the engine's bound or below its preset", s.MaxFPS)
		}
	}
	for _, b := range s.Beside {
		if b.Set == "" || b.Set == s.Name || b.Option == "" || len(b.Lease) == 0 || b.Commit == 0 {
			return fmt.Errorf("the PyTorch model set's figures beside %q are incomplete", b.Set)
		}
		for value, lease := range b.Lease {
			if value == "" || lease <= s.Lease {
				return fmt.Errorf("the PyTorch model set's lease beside %s is not above its own", b.Set)
			}
		}
	}
	for _, size := range s.Sizes {
		if size[0] < 32 || size[1] < 32 || size[0]%32 != 0 || size[1]%32 != 0 || size[0] > 1280 || size[1] > 1280 {
			return fmt.Errorf("the PyTorch model set's size %dx%d is not a multiple of 32 within 1280", size[0], size[1])
		}
	}
	switch s.FirstFrame {
	case FirstFrameOptional, FirstFrameRequired, FirstFrameNone:
	default:
		return fmt.Errorf("the PyTorch model set's first frame %q is none of optional, required and none", s.FirstFrame)
	}
	for _, o := range s.Options {
		if o.Name == "" || len(o.Values) == 0 {
			return fmt.Errorf("the PyTorch model set's option %q has no values", o.Name)
		}
	}
	return nil
}

// File is the file of the set passed with role.
func (s TorchModelSet) File(role string) (VideoModelFile, bool) {
	for _, f := range s.Files {
		if f.Role == role {
			return f, true
		}
	}
	return VideoModelFile{}, false
}

// TorchModelPlaces is the record of where each pinned file of a PyTorch set
// was found, and TorchModelRecord the metadata record of their placed proof,
// both in the provider's private state and apart from an sd-server set of
// the same name.
func TorchModelPlaces(stateDir string, set TorchModelSet) string {
	return filepath.Join(stateDir, "models-torch-"+set.Name+".places")
}

// TorchModelRecord is the metadata record of a PyTorch set's placed files.
func TorchModelRecord(stateDir string, set TorchModelSet) string {
	return filepath.Join(stateDir, "models-torch-"+set.Name+".files")
}
