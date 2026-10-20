package torch

import (
	"context"

	inference "github.com/openabstractions/abstraction-inference/go"
	inferenceservice "github.com/openabstractions/abstraction-inference/go/service"
	router "github.com/openabstractions/abstraction-router/go"
)

// ServiceHost serves the provider's profiles on one endpoint, binding every
// caller by native Program proof and refusing any program outside its
// runtime designation.
type ServiceHost = inferenceservice.Host

// Admit is the inference.Admitter this provider runs after authority and
// request preflight: the image set loads itself, under the holder, once the
// image runs (PYTORCH-T11), and any other model is not served here.
func Admit(h *Host) inference.Admitter {
	return func(_ context.Context, _ *router.Host, name string) (string, bool) {
		if h.ServesImage(name) {
			return "", true
		}
		return "model:undeclared", false
	}
}

// ProviderConfig is the inference provider configuration pytorchd serves:
// the image set as the provider's image engine for its own router host, the
// runtime's decision standing for the caller's rights, and Admit.
func ProviderConfig(h *Host, report func(error)) inference.Config {
	var images inference.ImageEngine
	var imageHost *router.Host
	if h != nil {
		images, imageHost = h.ImageEngine()
	}
	return inference.Config{
		// The image set serves its router host in place of an HTTP
		// adapter; its results need Config.PrepareContentWrite.
		ImageEngine:     images,
		ImageEngineHost: imageHost,
		Router:          router.New(),
		// The runtime bound the application and decided its rules before it
		// forwarded; this endpoint admits that runtime's account alone.
		Decide: func(context.Context, inference.Subject, string, string) (string, error) { return "permitted", nil },
		Apply: func(context.Context, inference.Subject, string, string, string) (map[string]string, string) {
			return nil, "applied"
		},
		Admit:   Admit(h),
		OnError: report,
	}
}

// Listen binds endpoint for the provider's inference provider as a one-hop
// leaf: calls admitted here use the local engine directly and refuse another
// native or remote OA delegation.
func Listen(endpoint string, provider *inference.Provider, runtimePrograms ...string) (*ServiceHost, error) {
	return inferenceservice.ListenDesignatedLeaf(endpoint, provider, inference.ExecutionLocal, runtimePrograms...)
}

// ImageOutputLimit is the bound of one published image: 20 MiB, for an image
// of any type; no other profile's output is published here.
func ImageOutputLimit(profile, media string) (int64, bool) {
	if profile == "image" && media == "image/*" {
		return 20 << 20, true
	}
	return 0, false
}
