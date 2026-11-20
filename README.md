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
7. SSH/SCP keep strict host-key checks and now force a compatible modern KEX set from local OpenSSH support to avoid provider KEX mismatches.

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

During `upscaler upscale`, the CLI now streams worker progress when the worker image emits `VAST_PROGRESS_JSON` events:

- Stage transitions (`preparation`, `decoding`, `upscale`, `encoding`, `finalization`)
- Frame progress (`frames_done/frames_total`) when available
- Measured processing FPS and ETA when computable
- Upload/download transfer start and completion markers

If output is redirected, progress is throttled to periodic readable lines instead of terminal in-place refresh.

Other operational commands:

```powershell
upscaler instances --config "config.yaml"
upscaler destroy 123456 --config "config.yaml" --yes
upscaler session resume --session-id sess-<id> --config "config.yaml" --known-hosts "$HOME\.ssh\known_hosts"
upscaler session cleanup --config "config.yaml"
upscaler status
upscaler history --limit 20
```

## Local web UI (Phase 5A V1)

Install dependencies (including FastAPI/Uvicorn), then start the local web interface:

```powershell
pip install -e .
upscaler web --config "config.yaml" --host 127.0.0.1 --port 8765
```

Open: `http://127.0.0.1:8765`

V1 behavior:

- Local-only bind: `127.0.0.1` (refused otherwise).
- SSE live stream for provisioning and worker progress events.
- One active processing task at a time.
- Automatic mode: keeps existing paid confirmation, provisioning, processing, and verified cleanup flow.
- Manual mode: uses an existing instance id, never auto-creates and never auto-destroys; billing continues until manual destroy.
- A prompt is single-use: each SSH/paid confirmation prompt id can be answered only once, and resolved prompts are not re-opened after SSE replay/reconnect.
- SSH trust checks remain strict/TOFU as in CLI via explicit browser confirmations.
- Automatic mode rejects placeholder Docker tags (for example `ghcr.io/acme/worker:test`); if no known real tag exists, enter one explicitly.
- No secrets are exposed to browser payloads.
- Existing output/report files are never overwritten.
- Main workflow is now organized in 4 steps:
  1. Choose video (from `<project-root>/video/`)
  2. Analyze locally (FFprobe + visual diagnostics such as cropdetect/idet)
  3. Validate preprocessing choices
  4. Configure and launch upscale
- Video sources are listed from `<project-root>/video/` (project-root derived from package location, not process CWD).
- Explicit source exclusion rule in video listing: files ending with `-upscaled.<ext>`, `-cropped.<ext>`, and `*.report.json` are hidden from source selection.
- Path safety for source selection: no path traversal, no path separators in source names, and no resolved path outside the allowed `video/` directory.
- Manual input path remains available under Advanced options for compatibility/troubleshooting.
- `Preprocessing plan path (optional)` remains available under Advanced options.
- If both a generated in-UI preprocessing plan and a manual preprocessing plan path are provided, the manual path takes precedence.
- In this phase, only crop is executed remotely; other recommendations (deinterlace/denoise/deblock) are displayed as disabled/advisory in UI.
- Interface ergonomique (phase 5F):
  - entièrement en français, sections visuelles par étape, largeur max adaptée aux écrans de bureau
  - recommandations présentées en cartes lisibles (diagnostic, recommandation, confiance, état)
  - commandes FFmpeg visibles uniquement dans des détails techniques repliables
  - recadrage avec boutons radio adjacents au libellé et désactivation visuelle des marges manuelles hors mode personnalisé
  - supervision enrichie: état courant, étape, progression (si fournie), durée, résultat, erreurs, journal SSE repliable

Validation manuelle rapide de l'IHM:

1. Démarrer l’UI locale puis ouvrir `http://127.0.0.1:8765`.
2. Étape 1: choisir une vidéo du dossier `video/` (vérifier nom, taille, résolution, durée, FPS).
3. Étape 2: lancer/attendre l’analyse locale (statut + résumé).
4. Étape 3: vérifier les recommandations, changer le mode crop, confirmer les dimensions finales (issues du plan validé).
5. Étape 4: choisir mode Vast.ai automatique ou manuel, vérifier le message contextuel du mode.
6. Lancer un job, vérifier le suivi (état, étape, progression si disponible, durée, résultat), tester annulation et prompts de confirmation.

Produced files and local state:

- Output video default: `<input-dir>/<input-stem>-upscaled.<ext>`
- Output report default: `<output-video>.report.json`
- Session store: `%USERPROFILE%\.vast_upscaler\sessions.json`
- Provisioning history store: `%USERPROFILE%\.vast_upscaler\provisioning_history.json`
- Workflow lock: `%USERPROFILE%\.vast_upscaler\workflow.lock`
- After `running`, SSH endpoint reachability is probed with bounded retries before trust bootstrap (strict cap, no infinite retry).

Common errors and actions:

- `Session store is corrupted`: stop paid runs and reconcile active instances in Vast console first.
- `Warning: provisioning metrics were not saved ...`: workflow result is preserved, but local provisioning history needs manual repair.
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
- `data/preprocessing/`: local preprocessing plan JSON files (generated/kept locally, ignored by Git).
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
