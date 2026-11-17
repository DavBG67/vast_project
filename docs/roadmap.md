# Roadmap

## Current status

- Phase 0: completed (repository audit).
- Phase 1: completed in this iteration (local foundation).
- Phase 2: completed in this iteration (read-only Vast.ai integration).
- Phase 3: completed in this iteration (worker Docker + Real-ESRGAN pipeline).

## Next milestones

- Phase 4: end-to-end orchestration with upload/process/download/destroy.
- Phase 5: quality/performance improvements and richer profiles.

## Phase 3 delivered

- `worker/` runtime modules and Docker image definition.
- Real-ESRGAN integration for `balanced` profile.
- Worker report JSON and output validation safeguards.
- Worker-focused CPU tests and Docker smoke workflow.
- GHCR workflow with manual publication controls.

## Phase 2 delivered

- Strict read-only Vast API client.
- Offers discovery with filtering and ranking.
- Offline fixture mode.
- Cost estimation with explicit assumptions and unknowns.
- No paid operations implemented.

## Guardrails

- No paid operation without explicit user confirmation.
- No cloud intermediary for video data transfer.
- No secrets committed to repository.
