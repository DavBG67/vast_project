# Models and Licenses (Phase 3)

## Initial model

- Engine: Real-ESRGAN
- Profile implemented: `balanced`
- Model weights: `RealESRGAN_x4plus.pth`
- Build-time model location inside image: `/opt/models/RealESRGAN_x4plus.pth`

## Dependency versions (worker)

Pinned in `worker/requirements.txt`:

- realesrgan==0.3.0
- basicsr==1.4.2
- torch==2.4.1
- torchvision==0.19.1
- opencv-python-headless==4.10.0.84
- numpy==1.26.4

## CUDA compatibility intent

Image base:

- `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`

Target GPUs:

- RTX 3090
- RTX 3090 Ti
- RTX 4090

Expected requirement:

- NVIDIA driver compatible with CUDA 12.1 runtime.

## License notes

- License metadata checked from PyPI for pinned versions:
	- realesrgan 0.3.0: BSD-3-Clause
	- basicsr 1.4.2: Apache-2.0
	- torch 2.4.1: BSD-3-Clause
	- torchvision 0.19.1: BSD
	- opencv-python-headless 4.10.0.84: Apache-2.0
- Real-ESRGAN model weights are downloaded from the official release artifact at build time.
- Transitive dependency licenses and model redistribution terms still require release-time compliance review.
- This repository does not embed any API keys or secrets in the worker image.

## Security note

PyPI metadata for some ML packages includes known vulnerability advisories in various historical versions.
Before production rollout, perform a security review and consider upgrading pinned versions when validated.

## Operational note

Only `balanced` is implemented in phase 3.

Profiles `fast` and `quality` intentionally fail with explicit error to avoid fake behavior.
