# Vast Video Upscaler

Python CLI orchestrator to inspect and prepare AI upscaling workflows on Vast.ai,
with strict financial and security guardrails.

This repository currently implements phases 1, 2, and 3:

- Packaging and CLI.
- Config loading and validation.
- ffprobe-based video inspection.
- Local session state readout.
- Vast.ai read-only offer discovery.
- Offer filtering and ranking.
- Cost estimation with explicit assumptions.
- Offline fixture mode for tests and demos.
- Worker Docker image with Real-ESRGAN pipeline (phase 3).
- Unit tests.

No paid Vast.ai operation is executed in this phase.

## Current CLI commands

- `upscaler doctor`
- `upscaler inspect "D:\Videos\film.mkv"`
- `upscaler offers`
- `upscaler offers --fixture tests/fixtures/offers.json`
- `upscaler estimate "D:\Videos\film.mkv" --fps-processing 5`
- `upscaler status`
- `upscaler history`

The following commands exist as safe placeholders and return "not implemented":

- `upscaler benchmark "D:\Videos\extrait.mkv"`
- `upscaler upscale "D:\Videos\film.mkv"`
- `upscaler instances`
- `upscaler destroy INSTANCE_ID`

## Worker (Phase 3)

Build worker image:

```powershell
docker build -f worker/Dockerfile -t vast-video-upscaler-worker:dev .
```

Smoke test (no GPU required):

```powershell
docker run --rm vast-video-upscaler-worker:dev --help
```

Expected GPU run command:

```powershell
docker run --rm --gpus all ^
	-v D:\Videos\input:/workspace/input ^
	-v D:\Videos\output:/workspace/output ^
	vast-video-upscaler-worker:dev ^
	--input /workspace/input/source.mkv ^
	--output /workspace/output/result.mkv ^
	--target-width 1920 ^
	--target-height 1080 ^
	--profile balanced
```

Phase 3B references:

- CI/GHCR policy: `docs/worker.md`
- GHCR + first GPU test runbook (procedure only): `docs/ghcr-and-gpu-test.md`

Manual publish workflow (do not run without approval):

- GitHub Actions workflow: `.github/workflows/docker-worker.yml`
- Input `publish=false` validates only (no push)
- Input `publish=true` pushes immutable tag `dev-sha-<commit>` to GHCR

## Prerequisites (Windows)

- Python 3.11+
- FFmpeg and ffprobe available in `PATH`
- PowerShell

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Configuration

1. Copy `config.example.yaml` to `config.yaml` and adjust values.
2. Copy `.env.example` to `.env` (or define env vars directly in your shell).
3. Set secrets only through environment variables:
	- `VAST_API_KEY`
	- `VAST_SSH_PRIVATE_KEY`

Never commit `config.yaml`, `.env`, private keys, or API tokens.

## Quick start

```powershell
upscaler doctor
upscaler inspect "D:\Videos\film.mkv"
upscaler offers
upscaler estimate "D:\Videos\film.mkv" --fps-processing 5
```

Offline mode (no Vast API call):

```powershell
upscaler offers --fixture tests/fixtures/offers.json
upscaler estimate "D:\Videos\film.mkv" --fixture tests/fixtures/offers.json --fps-processing 5
```

Compatibility entrypoint for planned usage style:

```powershell
python -m src.cli doctor
python -m src.cli inspect "D:\Videos\film.mkv"
```

## Testing

```powershell
pytest
```

## Repository layout

- `.github/agents/`: local multi-agent prompts.
- `docs/`: architecture, security, roadmap, ADR, specs.
- `public/`: web public folder (unused in phase 1).
- `scripts/`: helper scripts.
- `src/vast_upscaler/`: Python package.
- `tests/`: unit tests.

## Limits in phase 1

Phase 3 limits and safeguards:

- Client only uses read-only endpoints.
- No create/stop/start/destroy instance path is implemented.
- Cost estimation is hypothesis-driven and can be partial.
- No benchmark-derived processing speed is available yet.
- Worker CI is CPU-only (no paid GPU validation in CI).
- CUDA execution must be validated on a real NVIDIA environment.
