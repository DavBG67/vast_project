# Vast Video Upscaler

Python CLI orchestrator to inspect and prepare AI upscaling workflows on Vast.ai,
with strict financial and security guardrails.

This repository currently implements phases 1, 2, 3, 4A, 4B, 4C, 4D, and 4E:

- Packaging and CLI.
- Config loading and validation.
- ffprobe-based video inspection.
- Local session state readout.
- Vast.ai read-only offer discovery.
- Offer filtering and ranking.
- Cost estimation with explicit assumptions.
- Offline fixture mode for tests and demos.
- Worker Docker image with Real-ESRGAN pipeline (phase 3).
- Instance lifecycle orchestration (phase 4E).
- SSH/SCP transfer with strict host-key policy (phase 4C).
- Remote worker execution and artifact validation (phase 4D).
- Unit tests.

Paid operations are possible only through explicit user confirmation in `upscale` real mode.

## Current CLI commands

- `upscaler doctor`
- `upscaler inspect "D:\Videos\film.mkv"`
- `upscaler offers`
- `upscaler offers --fixture tests/fixtures/offers.json`
- `upscaler estimate "D:\Videos\film.mkv" --fps-processing 5`
- `upscaler status`
- `upscaler history`

The following command remains a placeholder:

- `upscaler benchmark "D:\Videos\extrait.mkv"`

Phase 4E commands:

- `upscaler upscale "D:\Videos\film.mkv" --worker-image ghcr.io/<owner>/vast-video-upscaler-worker:dev-sha-<sha>`
- `upscaler upscale "D:\Videos\film.mkv" --worker-image <image> --dry-run`
- `upscaler instances`
- `upscaler destroy 123456 --yes`
- `upscaler session resume --session-id sess-<id>`
- `upscaler session cleanup`

Detailed docs:

- Integration audit: `docs/phase4-integration-audit.md`
- Real-test runbook (do not execute without approval): `docs/runbook-first-real-test.md`
- Security controls: `docs/security.md`

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
- OpenSSH client tools in `PATH`: `ssh`, `scp`, `ssh-keyscan`
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
4. The CLI auto-loads a local `.env` file from the project root if present; existing system environment variables keep priority.

Never commit `config.yaml`, `.env`, private keys, or API tokens.

Required runtime parameters for real orchestration:

- API key: `VAST_API_KEY`
- SSH key path: `VAST_SSH_PRIVATE_KEY`
- Worker image: CLI option `--worker-image`
- Known-hosts file: CLI option `--known-hosts` (default: `%USERPROFILE%\.ssh\known_hosts`)

Initial SSH trust bootstrap (operator-assisted):

1. Obtain host/port from workflow output.
2. The CLI runs `ssh-keyscan` and shows the ED25519 SHA256 fingerprint.
3. By default (`--ssh-bootstrap-mode tofu`), accepting this key is Trust On First Use (TOFU): it is not independently verified by Vast.ai API/UI.
4. Confirm explicitly to append the key to `known_hosts`.
5. If an approved key exists and a different fingerprint is observed, the workflow is refused.
6. Optional strict mode (`--ssh-bootstrap-mode strict`) keeps explicit confirmation and also asks for an independently verified fingerprint.

## Quick start

```powershell
upscaler doctor
upscaler inspect "D:\Videos\film.mkv"
upscaler offers
upscaler estimate "D:\Videos\film.mkv" --fps-processing 5
```

Phase 4E dry-run example (no paid operation):

```powershell
upscaler upscale "D:\Videos\film.mkv" ^
	--worker-image "ghcr.io/<owner>/vast-video-upscaler-worker:dev-sha-<sha>" ^
	--dry-run ^
	--config "config.yaml" ^
	--known-hosts "$HOME\.ssh\known_hosts"
```

Phase 4E real run example (requires explicit confirmation):

```powershell
upscaler upscale "D:\Videos\film.mkv" ^
	--worker-image "ghcr.io/<owner>/vast-video-upscaler-worker:dev-sha-<sha>" ^
	--offer-id 31129 ^
	--provisioning-timeout-seconds 600 ^
	--config "config.yaml" ^
	--known-hosts "$HOME\.ssh\known_hosts"
```

Other operational commands:

```powershell
upscaler instances --config "config.yaml"
upscaler destroy 123456 --config "config.yaml" --yes
upscaler session resume --session-id sess-<id> --config "config.yaml" --known-hosts "$HOME\.ssh\known_hosts"
upscaler session cleanup --config "config.yaml"
upscaler status
upscaler history --limit 20
```

Produced files and local state:

- Output video default: `<input-dir>/<input-stem>-upscaled.<ext>`
- Output report default: `<output-video>.report.json`
- Session store: `%USERPROFILE%\.vast_upscaler\sessions.json`
- Workflow lock: `%USERPROFILE%\.vast_upscaler\workflow.lock`

Common errors and actions:

- `Session store is corrupted`: stop paid runs and reconcile active instances in Vast console first.
- `SSH trust bootstrap required`: validate fingerprint manually, update known_hosts, then retry.
- `Destroy requested but disappearance not verified`: run `upscaler instances`, then `upscaler destroy <id> --yes`, then `upscaler session cleanup`.
- `Cancelled before paid operation`: rerun and confirm only if estimate/risk is acceptable.

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

Phase 4E safeguards:

- Paid creation requires explicit confirmation.
- `--dry-run` never creates or destroys instances.
- Strict SSH host key checking remains enabled.
- Local timeout is not a billing ceiling guarantee.
- Cleanup verifies instance disappearance; API success alone is insufficient.
- Ambiguous creation/processing outcomes are preserved for manual reconciliation.
- Provisioning progress is printed during readiness wait (instance id, GPU, host, status, elapsed time).
- `--offer-id` enforces explicit offer selection; no silent fallback to another offer.
- Provisioning timeout is configurable via `--provisioning-timeout-seconds` (default 600s).
- Cost estimation is hypothesis-driven and can be partial.
- No benchmark-derived processing speed is available yet.
- Worker CI remains CPU-only (no paid GPU validation in CI).

Financial note:

- Max session duration and hourly price constraints are guardrails, not absolute billing guarantees.
