# GHCR and First GPU Test (Preparation Only)

This document prepares manual publication and the first real GPU validation run.
No publication and no Vast.ai instance action is executed here.

## GHCR package visibility and naming

- Registry: `ghcr.io`
- Image: `ghcr.io/<owner>/vast-video-upscaler-worker`
- Published tag (manual workflow only):
  - `dev-sha-<12-char-commit>`

Recommended visibility for first tests:

- Keep package private if your Vast.ai workflow can authenticate to GHCR.
- Switch to public only if needed for simpler pull behavior on test instances.

## Manual publication trigger (do not run without approval)

Workflow: `.github/workflows/docker-worker.yml`

Inputs:

- `publish`:
  - `false` (default): validation-only, no push.
  - `true`: push GHCR tag `dev-sha-<commit>`.

## Vast.ai first GPU test runbook (procedure only)

Goal: validate real CUDA execution and output quality on a short sample.

### Prerequisites

- GHCR image exists (manual publish already approved and executed).
- Test clip ready (10-30 seconds, representative content).
- SSH key configured for Vast.ai instance access.
- Cost/time guardrails pre-defined.

### Steps

1. Select image on Vast.ai
- Use custom image: `ghcr.io/<owner>/vast-video-upscaler-worker:dev-sha-<commit>`.
- Ensure instance has NVIDIA GPU and enough disk.

2. Start instance with SSH access
- Enable SSH.
- Wait for instance readiness.

3. Validate GPU on instance
- Run `nvidia-smi`.
- Confirm driver and GPU are visible.

4. Validate CUDA in PyTorch
- Run:
  - `python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"`
- Expected:
  - `torch.cuda.is_available()` must be `True`.

5. Transfer input clip
- Upload 10-30 sec source clip to `/workspace/input/source.mkv`.

6. Run worker
- Command:
  - `python process.py --input /workspace/input/source.mkv --output /workspace/output/result.mkv --target-width 1920 --target-height 1080 --profile balanced`

7. Retrieve outputs
- Download:
  - `/workspace/output/result.mkv`
  - `/workspace/output/result.mkv.report.json`

8. Validate result
- Check:
  - command exit code is zero,
  - report status is `completed`,
  - output duration/frame cadence are coherent,
  - visual quality is acceptable.

9. Destroy instance
- Terminate instance immediately after validation.

## Fallback procedure and duration cap

Duration cap recommendation for first run:

- Hard cap: 30 minutes wall-clock from instance ready to destroy.

If failure occurs:

1. Capture diagnostics before destroy:
- worker stderr/stdout,
- report JSON,
- `nvidia-smi`,
- `python -c` CUDA check output,
- free disk (`df -h`).

2. Classify failure:
- Image pull/auth issue,
- CUDA unavailable or mismatch,
- OOM during inference,
- ffmpeg/muxing error,
- model load error.

3. Stop and destroy instance.

4. Apply fix in repository and rerun validation workflow before another GPU attempt.
