# Worker Docker (Phase 3)

## Scope

Phase 3 adds a standalone Linux worker container for AI upscaling with Real-ESRGAN on NVIDIA CUDA.

This phase does not create or manage Vast.ai instances.

## Worker entrypoint

Command:

```bash
python process.py \
  --input /workspace/input/source.mkv \
  --output /workspace/output/result.mkv \
  --target-width 1920 \
  --target-height 1080 \
  --profile balanced
```

## Pipeline

1. Inspect source with ffprobe.
2. Decode video frames in stream mode (rawvideo pipe).
3. Upscale each frame with Real-ESRGAN on CUDA.
4. Fit to target canvas with letterboxing (no implicit crop/stretch).
5. Encode video-only HEVC stream.
6. Mux original audio/subtitles/metadata when compatible.
7. Validate output and atomically rename temporary file.
8. Write JSON report.

## Robustness controls

- SIGINT/SIGTERM handling.
- Temporary workspace cleanup.
- Non-zero exit on failures.
- FFmpeg decode/encode error propagation.
- CUDA and OOM errors surfaced as explicit failures.
- Output validation before final rename.

## CPU-only validation limits

CI validates:

- Python unit tests.
- Docker image build.
- Container smoke command (`--help`).

CI does not validate:

- Real GPU inference correctness/performance.
- End-to-end CUDA execution on RTX 3090/3090 Ti/4090.

A real GPU run is required for final CUDA validation.

## Phase 3B CI validation

Workflow: `.github/workflows/docker-worker.yml`

Validation scope on GitHub Actions (no GPU required):

1. `ruff check .`
2. `pytest`
3. Docker build `linux/amd64`
4. Entrypoint smoke run (`--help`)
5. FFmpeg/ffprobe presence checks
6. Python imports and runtime version print for `torch`, `torchvision`, `realesrgan`, `basicsr`, `cv2`
7. Real-ESRGAN weights SHA256 verification in image
8. Image size report in step summary

Important:

- `torch.cuda.is_available()` in this workflow is expected to be `False` (no GPU runner).
- This workflow does not claim CUDA runtime validation on NVIDIA hardware.

## GHCR publication policy

- Publication is manual only (`workflow_dispatch` with `publish=true`).
- Publish job has minimal required permission: `packages: write` and `contents: read`.
- Tag policy: immutable development tag linked to commit SHA:
  - `ghcr.io/<owner>/vast-video-upscaler-worker:dev-sha-<12-char-commit>`
- `latest` is intentionally not published at this stage.
