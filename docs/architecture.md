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

## Visual diagnostics (local, optional, non-destructive)

- Module: [source_analysis.py](../src/vast_upscaler/source_analysis.py)
- Trigger:
  - CLI: `upscaler analyze-source <video> --visual-diagnostics`
  - API: `POST /api/video/analyze` with `include_visual_diagnostics=true`
- Existing default contracts are preserved:
  - `upscaler analyze-source <video>` (without option) keeps metadata-only output.
  - `POST /api/video/analyze` without `include_visual_diagnostics` keeps metadata-only output.

### Report structure

- Top-level fields: ffprobe-derived metadata (certain when present).
- Optional `analysis_report`:
  - `version`: report schema version (`1.0`)
  - `metadata_certainty`: explains certainty boundary for metadata
  - `heuristics_certainty`: explains certainty boundary for diagnostics
  - `visual_diagnostics`:
    - `sampling_points_seconds`
    - `options` (sample count, duration, max analyzed duration, ffmpeg threads, timeout, min samples)
    - `cropdetect` section (recommended crop + confidence + inconsistencies)
    - `idet` section (progressive/interlaced/mixed/indeterminate + confidence)
    - `noise_artifacts` section (heuristic indicators and conservative confidence)
    - `partial` flag when any diagnostic section partially fails

### Reliability limits

- `cropdetect`: can be perturbed by very dark scenes, fades, subtitles, and variable aspect-ratio content.
- `idet`: classification confidence drops when sampled segments are too short or inconsistent.
- `noise_artifacts`: first version is intentionally conservative and can return `unknown`.
- Any filter failure returns partial diagnostics rather than hard-failing metadata analysis.

### Examples

CLI metadata only:

```bash
upscaler analyze-source /path/to/video.mp4
```

CLI with visual diagnostics:

```bash
upscaler analyze-source /path/to/video.mp4 \
  --visual-diagnostics \
  --sample-count 5 \
  --sample-duration-seconds 1.5 \
  --max-analyzed-duration-seconds 180 \
  --ffmpeg-threads 1 \
  --ffmpeg-timeout-seconds 30 \
  --min-samples-required 3
```

API metadata only:

```http
POST /api/video/analyze
Content-Type: application/json

{"input_path":"C:/videos/input.mp4"}
```

API with visual diagnostics:

```http
POST /api/video/analyze
Content-Type: application/json

{
  "input_path":"C:/videos/input.mp4",
  "include_visual_diagnostics": true,
  "diagnostic_options": {
    "sample_count": 5,
    "sample_duration_seconds": 1.5,
    "max_analyzed_duration_seconds": 180.0,
    "ffmpeg_threads": 1,
    "ffmpeg_timeout_seconds": 30.0,
    "min_samples_required": 3
  }
}
```

## Intelligent preprocessing planner (recommendation only)

- Module: [preprocessing_planner.py](../src/vast_upscaler/preprocessing_planner.py)
- Scope:
  - **Analyser → Recommander → Choisir** only.
  - No preprocessing execution in this phase.
  - No Vast.ai dependency.

### Inputs

- Existing analysis payload JSON (metadata only or metadata + visual diagnostics), or
- A fresh analysis generated by:
  - CLI video path mode, or
  - API `input_path` mode.

If a valid analysis payload is supplied, planner does not rerun FFmpeg diagnostics.

### Planning rules

- Crop:
  - Default mode `preserve` (disabled operation).
  - `auto_crop` requires explicit user selection **and** usable crop diagnostics confidence.
  - `custom_crop` requires validated margins and in-bounds rectangle.
  - Stretching is always disallowed (`allow_stretch=false`).
- Deinterlace:
  - Never auto-enabled.
  - Recommended only when `idet` is sufficiently convincing.
- Denoise / Deblock:
  - Conservative recommendations only.
  - `unknown` is allowed and preferred over overconfident output.

### Plan JSON schema (logical)

- `version`
- `source_summary`
- `recommendations`
- `user_choices`
- `operations`
- `warnings`
- `estimated_impact`

Each operation includes:

- stable `operation_id`
- `enabled` (selected by user, never implicitly activated)
- `recommended` (planner advice)
- `parameters` (validated and explicit)
- `justification` + `risk_notes`

`estimated_impact` is qualitative only (`low|medium|high|unknown`) and never uses acceleration percentages.

### Interfaces

CLI:

```bash
upscaler plan-preprocessing <video_path>
upscaler plan-preprocessing <video_path> --visual-diagnostics
upscaler plan-preprocessing --analysis-report C:/path/to/analysis.json
upscaler plan-preprocessing <video_path> --crop-mode custom_crop --crop-top 8 --crop-bottom 8 --crop-left 0 --crop-right 0
```

API:

```http
POST /api/video/preprocessing-plan
Content-Type: application/json

{
  "analysis_payload": { "...": "existing analyze-source payload" },
  "user_choices": {
    "crop_mode": "preserve",
    "deinterlace_mode": "preserve",
    "denoise_mode": "preserve",
    "deblock_mode": "preserve"
  }
}
```

Alternative API sources:

- `input_path` (+ optional `include_visual_diagnostics` and `diagnostic_options`)
- `analysis_report_path`

Exactly one source is accepted per request.
