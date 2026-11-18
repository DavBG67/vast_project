# Phase 4J — Support NVIDIA Blackwell / RTX 5090 (Preparation)

## Scope and status

- This phase prepares an **experimental** Blackwell-capable worker variant.
- No Vast.ai rental, no paid processing, and no image publication are executed in this phase.
- Stable image/tag remains unchanged: `ghcr.io/davbg67/vast-video-upscaler-worker:dev-sha-4ef981595f50`.

## Version references used

The selected stack is based on public compatibility references:

1. PyTorch previous versions matrix (torch/torchvision/cuda index URLs):
   - https://pytorch.org/get-started/previous-versions/
2. Official `pytorch/pytorch` Docker tags:
   - https://hub.docker.com/r/pytorch/pytorch/tags

From these references, this phase uses:

- `torch==2.14.1`
- `torchvision==0.29.1`
- CUDA runtime compiled in torch: `13.0` (>= 12.8 requirement)
- Base image: `pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime`

## Delivered artifacts

1. Experimental Docker variant:
   - [worker/Dockerfile.blackwell](../worker/Dockerfile.blackwell)
   - [worker/requirements.blackwell.txt](../worker/requirements.blackwell.txt)
2. Worker startup CUDA diagnostics (logs only; no secret exposure):
   - [worker/process.py](../worker/process.py)
3. CI workflow prepared for alternate Dockerfile/version checks and optional experimental tag:
   - [docker-worker.yml](../.github/workflows/docker-worker.yml)

## PEP 668 compatibility (GitHub Actions build)

- Base image Python can be externally managed (PEP 668), making global `pip install --upgrade pip` fail.
- `Dockerfile.blackwell` now performs runtime interpreter diagnostics (`python` / `python3`) and auto-selects the first interpreter that can import preinstalled `torch`.
- It then creates a dedicated venv with `--system-site-packages` from that detected interpreter so preinstalled torch/CUDA remain available without reinstalling torch.
- All install commands run through the same environment (`python -m pip` with `PATH` preferring `${WORKER_VENV}/bin`).

## Compatibility table (prepared, not hardware-validated)

| Component | Stable stack | Blackwell experimental stack |
|---|---:|---:|
| Base image | `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` | `pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime` |
| torch | 2.4.1 | 2.14.1 |
| torchvision | 0.19.1 | 0.29.1 |
| realesrgan | 0.3.0 | 0.3.0 |
| basicsr | 1.4.2 | 1.4.2 |
| CUDA compiled in torch | 12.1 | 13.0 |

## Real-ESRGAN / BasicSR compatibility handling

- Runtime behavior is preserved (decode → upscale → encode → report).
- Existing BasicSR import patch logic is retained in `Dockerfile.blackwell` for torchvision path changes.
- FP16 inference path remains enabled (`half=True`) exactly as in the current worker pipeline.

## Startup CUDA diagnostics

The worker now logs, at startup:

- PyTorch version
- CUDA version compiled into PyTorch
- `torch.cuda.is_available()`
- GPU name
- Compute capability
- Torch-supported arch list (when available)

If CUDA is available but current GPU capability is absent from torch arch list, worker raises a clear `WorkerError` before processing starts.

This does **not** modify the JSON progress event protocol (`VAST_PROGRESS_JSON ...`) consumed by the orchestrator.

## CI/build publication preparation

Workflow [docker-worker.yml](../.github/workflows/docker-worker.yml) now supports manual inputs:

- `dockerfile` (default `worker/Dockerfile`)
- `expected_torch_version`
- `expected_torchvision_version`
- `expected_cuda_version`
- `publish_tag` (optional extra tag, e.g. `blackwell-test`)

Default behavior remains stable path-compatible.

### Example manual workflow dispatch for experimental image (procedure only)

- `publish=false` for validation-only build.
- `dockerfile=worker/Dockerfile.blackwell`
- `expected_torch_version=2.14.1`
- `expected_torchvision_version=0.29.1`
- `expected_cuda_version=13.0`

To publish after explicit approval:

- `publish=true`
- optionally `publish_tag=blackwell-test`

The workflow will always keep `dev-sha-<commit>` and optionally add `:blackwell-test`; it does not overwrite stable tags.

## Progressive validation protocol (for later paid test approval)

1. Run local lint/tests.
2. Run CI smoke build using `worker/Dockerfile.blackwell`.
3. Inspect runtime info step output (torch/torchvision/cuda versions).
4. Publish experimental tag only after explicit approval.
5. Rent **manual** RTX 5090 instance.
6. Validate:
   - `nvidia-smi`
   - `python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print(getattr(torch.cuda,'get_arch_list',lambda:[])())"`
7. Run Real-ESRGAN on one image.
8. Run 5–10s video upscale.
9. Verify:
   - output video
   - report JSON status/metrics
   - progress events
10. Destroy instance manually.

## Risks before first paid Blackwell run

- Prepared stack is version-coherent, but **hardware validation on RTX 5090 is still required**.
- Vast host driver/runtime mismatches can still fail CUDA runtime even with a newer container.
- Real throughput/VRAM behavior vs RTX 4090 remains unknown until benchmark.

## Clear decision

- **Do not claim RTX 5090 compatibility as validated yet.**
- Compatibility is **prepared** by stack/version alignment and diagnostics.
- First real compatibility verdict requires an explicit manual RTX 5090 run.
