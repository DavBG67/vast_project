# ADR 0002 - Initial GPU worker with Real-ESRGAN

## Status

Accepted

## Context

Phase 3 requires a real AI upscaling worker container that can run on Vast.ai NVIDIA GPUs.
The worker must be reproducible, avoid paid operations, and provide a concrete pipeline before adding more advanced engines.

## Decision

Adopt Real-ESRGAN as the first integrated GPU upscaler with a single supported profile:

- Profile: `balanced`
- Model: `RealESRGAN_x4plus`
- Runtime: PyTorch CUDA base image
- Processing approach: frame streaming via FFmpeg rawvideo pipes
- Output strategy: temporary output + validation + atomic rename

## Consequences

Positive:

- Delivers an actual AI upscaling path instead of synthetic placeholders.
- Keeps architecture simple and testable before adding VapourSynth/vs-mlrt/TensorRT.
- Supports future extension by adding new pipeline modules.

Trade-offs:

- Phase 3 CPU CI cannot prove CUDA runtime correctness.
- Image size is larger due to PyTorch and model artifacts.
- Initial implementation prioritizes reliability over maximal throughput.

## Alternatives considered

- FFmpeg-only scaling as temporary "AI" profile.
  - Rejected: does not satisfy real IA upscaling requirement.
- Integrating VapourSynth/vs-mlrt/TensorRT immediately.
  - Rejected: higher integration complexity and validation scope in phase 3.
