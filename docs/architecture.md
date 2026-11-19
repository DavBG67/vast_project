# Vast Video Upscaler Architecture (Phase 4E)

## Scope

Phases 1 to 4E deliver:

- Python package and CLI.
- Config loading and validation.
- Video inspection through ffprobe.
- Local session history store.
- Read-only Vast.ai offers discovery.
- Offer filtering, ranking, and simulation-based cost estimation.
- Standalone worker container for Real-ESRGAN GPU upscaling.
- Writable Vast instance lifecycle orchestration with explicit confirmation guardrails.
- Strict SSH/SCP transport with host-key verification.
- Remote worker execution and artifact recovery paths.

## Components

- `vast_upscaler.cli`: user-facing commands.
- `vast_upscaler.config`: YAML + environment settings.
- `vast_upscaler.video`: ffprobe integration and typed metadata.
- `vast_upscaler.session_store`: local JSON session records.
- `vast_upscaler.vast_client`: strict read-only HTTP client for Vast API.
- `vast_upscaler.offers`: offer query/filter/ranking logic independent from HTTP.
- `vast_upscaler.cost`: estimation logic with explicit unknowns and assumptions.
- `vast_upscaler.vast_write_client`: writable Vast API client for create/show/list/destroy.
- `vast_upscaler.instances`: lifecycle helpers (create/wait/destroy ownership checks).
- `vast_upscaler.transfer`: OpenSSH-based transfer and remote command execution.
- `vast_upscaler.remote_worker`: remote worker command + report/output validation.
- `vast_upscaler.orchestrator`: phase 4E workflow assembly and recovery/cleanup.
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

Implemented lifecycle model includes:

- `CREATED -> OFFER_SELECTED -> CONFIRMED -> INSTANCE_REQUESTED -> INSTANCE_CREATED -> INSTANCE_READY -> PROCESSING -> DOWNLOADING -> VERIFYING -> COMPLETED -> DESTROYED`
- Recovery/cleanup states: `FAILED`, `PRESERVED_FOR_RECOVERY`, `CLEANUP_PENDING`, `CLEANUP_REQUIRED`, `CANCELLED`

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

## Remaining extension points

- UX and policy hardening around recovery decisions.
- Expanded resume strategies for partially completed sessions.
- Real-world performance/cost calibration from benchmark data.

## Video profiles and local source analysis (Phase 4K kickoff)

### Audit conclusions (current behavior)

- Profile resolution path is config-driven and currently uses `video.default_profile` end-to-end:
  - CLI/web pass the configured profile into orchestration and remote worker command arguments.
  - Worker model selection still supports only `balanced` at runtime and fails explicitly for other values.
- Worker pipeline remains:
  1. ffprobe source probe
  2. ffmpeg decode of video stream (`rawvideo`, RGB24)
  3. Real-ESRGAN frame upscale (CUDA)
  4. ffmpeg encode (`libx265`, CRF, CPU encoding path)
  5. stream copy mux for audio/subtitles/metadata
- Existing local inspection was available but minimal; no local, non-destructive profile/preprocessing diagnosis API existed for the web app.

### Introduced architecture boundary

- New local-only analysis module: `vast_upscaler.source_analysis`
  - Inputs: file path + ffprobe payload.
  - Outputs: structured JSON with resolution, FPS, duration, codecs, bitrate, audio/subtitle tracks, interlace info.
  - Includes confidence distinction for interlace diagnosis (`certain` vs `uncertain`).
- New integration surfaces (without changing existing ones):
  - CLI: `upscaler analyze-source <video>`
  - Web: `POST /api/video/analyze` with `{"input_path":"..."}`

### Compatibility guarantees retained

- Balanced pipeline behavior is unchanged.
- Existing JSON contracts are preserved (`inspect`, existing job/config endpoints).
- No local destructive processing is performed by the analysis step.
- No Vast.ai operation is triggered by source analysis.

### Next steps

1. Add profile policy layer (`fast` / `balanced` / `quality` / `custom`) decoupled from worker model implementation.
2. Add optional preprocessing-plan layer (deinterlace/noise/crop decisions) from analysis facts only.
3. Introduce execution planner selecting local vs remote operations while preserving balanced defaults.
4. Add explicit final encoding policy object (codec, CRF/preset, mux policy) and compatibility tests.
