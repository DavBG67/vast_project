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

## Phase 3C post-GPU corrections

First real Vast.ai SSH test surfaced two issues that CI did not previously cover.

### 1) BasicSR / torchvision import compatibility

- Symptom: `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`.
- Root cause: `basicsr==1.4.2` imports `rgb_to_grayscale` from an old torchvision module path.
- Fix retained:
  - Build-time patch in Docker image for `basicsr/data/degradations.py`:
    - from `torchvision.transforms.functional_tensor`
    - to `torchvision.transforms.functional`
  - Build fails explicitly if expected patch target is not found.
  - Build also verifies the real import chain (`basicsr.data.degradations`, `RRDBNet`, `RealESRGANer`).

### 2) FFmpeg staged output path

- Symptom: `Unable to find a suitable output format for ... .mp4.tmp`.
- Root cause: temporary file path used `.suffix + .tmp` (`name.mp4.tmp`), so ffmpeg could not infer MP4 muxer from final extension.
- Fix retained:
  - Staged output now preserves container suffix: `name.tmp.mp4` (or `name.tmp.mkv`).
  - Atomic publish behavior is unchanged (rename staged file to final output after validation).
  - Staged file is cleaned up on failures.

### 3) Failure report metrics

- On late failure after frame processing (for example mux failure), report now keeps `frames_processed` and computes `processing_fps` from elapsed time.
- This avoids reporting a full-processing attempt as `frames_processed=0`.

## Additional CI checks (CPU-only)

Workflow: `.github/workflows/docker-worker.yml`

Added checks:

1. Targeted BasicSR/Real-ESRGAN import chain verification.
2. Temporary output naming verification (`.tmp.<ext>`).
3. Short CPU ffmpeg encode test + `ffprobe` verification.
4. Unit tests for JSON report correctness on late failures.

Limits unchanged:

- CI still does not validate CUDA runtime or GPU inference quality/performance.

## Vast.ai SSH retest procedure (post-fix)

1. Launch SSH instance with fixed image tag.
2. Validate `nvidia-smi` and `torch.cuda.is_available()`.
3. Upload short test clip to `/workspace/input/`.
4. Run worker command with explicit input/output paths.
5. Download output video and report JSON.
6. Verify status, frame count, duration/fps coherence, and visual output.
7. Destroy instance immediately.

## GHCR publication policy

- Publication is manual only (`workflow_dispatch` with `publish=true`).
- Publish job has minimal required permission: `packages: write` and `contents: read`.
- Tag policy: immutable development tag linked to commit SHA:
  - `ghcr.io/<owner>/vast-video-upscaler-worker:dev-sha-<12-char-commit>`
- `latest` is intentionally not published at this stage.
