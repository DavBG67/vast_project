# Vast Video Upscaler Architecture (Phase 3)

## Scope

Phases 1 to 3 deliver:

- Python package and CLI.
- Config loading and validation.
- Video inspection through ffprobe.
- Local session history store.
- Read-only Vast.ai offers discovery.
- Offer filtering, ranking, and simulation-based cost estimation.
- Standalone worker container for Real-ESRGAN GPU upscaling.
- No paid Vast.ai operation.

## Components

- `vast_upscaler.cli`: user-facing commands.
- `vast_upscaler.config`: YAML + environment settings.
- `vast_upscaler.video`: ffprobe integration and typed metadata.
- `vast_upscaler.session_store`: local JSON session records.
- `vast_upscaler.vast_client`: strict read-only HTTP client for Vast API.
- `vast_upscaler.offers`: offer query/filter/ranking logic independent from HTTP.
- `vast_upscaler.cost`: estimation logic with explicit unknowns and assumptions.
- `worker/process.py`: end-to-end worker runtime.
- `worker/video_probe.py`: ffprobe parsing and validation.
- `worker/pipelines/realesrgan.py`: Real-ESRGAN CUDA inference module.

## Execution flow in phase 3

1. User runs `upscaler doctor`.
2. CLI checks local prerequisites and config.
3. User runs `upscaler inspect <video>`.
4. CLI invokes ffprobe and prints normalized metadata.
5. User runs `upscaler offers`.
6. CLI builds strict read-only query for `POST /api/v0/bundles/` or uses fixture mode.
7. Offers are normalized, filtered by constraints, ranked, and displayed in a Rich table.
8. User runs `upscaler estimate <video> --fps-processing <value>`.
9. CLI computes a simulation estimate with explicit assumptions and unknown costs.
10. Worker image can process video via `process.py` using Real-ESRGAN on CUDA.

## Lifecycle status

The target lifecycle is documented but not yet fully executed:

- `CREATED -> STARTING -> READY -> UPLOADING -> PROCESSING -> DOWNLOADING -> VERIFYING -> COMPLETED -> DESTROYED`
- `FAILED`, `INTERRUPTED`, `CLEANUP_PENDING`

Full lifecycle automation is planned for phase 4.

## API boundaries and guardrails

- Allowed endpoints in phase 2:
	- `POST /api/v0/bundles/` (offer search)
	- `GET /api/v0/users/current/` (auth check utility)
- Any non-whitelisted endpoint is blocked by the client at runtime.
- Retries/backoff are applied only to idempotent read-only operations.
- Rate limiting follows API guidance for HTTP 429 without relying on `Retry-After`.

## Worker processing boundaries

- Worker performs local container processing only.
- Worker does not call Vast.ai APIs.
- Worker uses temporary output and final atomic rename after validation.
- Balanced profile is implemented; other profiles intentionally fail explicitly.
- CPU CI validates build and unit tests but does not validate CUDA throughput/quality.

## Extension points

The following modules are intentionally not active yet:

- `instances.py` (phase 4)
- `transfer.py` (phase 4)
- `orchestrator.py` (phase 4)
