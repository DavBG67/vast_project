# Phase 4 Integration Audit (A-E)

Date: 2026-11-17

Scope: static and simulated audit of implemented code paths across phase 4A to 4E.

No paid operation was executed during this audit.

## 1) Interface coherence across modules

- CLI -> orchestrator:
  - `upscale` builds `OrchestrationOptions` and delegates to `run_upscale_workflow`.
  - `instances` delegates to `list_active_instances`.
  - `destroy` delegates to `destroy_instance_with_verification`.
  - `session resume` delegates to `resume_session`.
  - `session cleanup` delegates to `cleanup_sessions`.
- Orchestrator -> instance lifecycle:
  - Uses `create_instance_with_confirmation` and `wait_until_instance_ready`.
- Orchestrator -> transport + remote worker:
  - Uses strict SSH connection builder from `transfer.py` and `execute_remote_worker` / `recover_remote_results`.
- Orchestrator -> session store:
  - Persists transitions after each critical step and preserves recovery/error metadata.

Audit verdict: coherent interfaces, no duplicate responsibility detected.

## 2) Vast.ai creation payload compatibility

Current payload (via `CreateInstanceRequest` in write client):

- HTTP method/path: `PUT /api/v0/asks/{ask_id}/`
- body fields used: `runtype`, `target_state`, `cancel_unavail`, `image`, `disk`, `label`, `env` (optional)

In orchestrator, current defaults are:

- `runtype="ssh_direct"`
- `target_state="running"`
- `cancel_unavail=True`
- `image` from CLI `--worker-image`
- `disk` from config `vast.disk_gb`
- `label` from generated session id

Audit verdict: payload is internally coherent and aligned with implemented write client contract.

Not fully validated without real test:

- acceptance of chosen runtype/image combination by live Vast API for selected offers
- provider-side edge behavior under API latency or partial fulfillment

## 3) GHCR image and SSH mode configuration

- Worker image is explicitly required by CLI (`--worker-image`).
- SSH mode is set through create payload (`ssh_direct`).
- SSH transport enforces strict host key checking and dedicated `known_hosts` usage.

Audit verdict: configuration path is explicit and wired end-to-end.

## 4) Worker parameter transmission

`remote_worker.py` builds:

- `python /app/worker/process.py`
- `--input`, `--output`, `--target-width`, `--target-height`, `--profile`, `--video-crf`, `--models-dir`, `--report`, `--min-free-gb`

Audit verdict: parameters required by worker process are transmitted.

## 5) Local/remote paths coherence

- Per-session remote root: `/workspace/upscale-sessions/<sanitized-session-id>/...`
- Remote files:
  - input: `input/<local filename>`
  - output: `output/<local output filename>`
  - report: `artifacts/report.json`
- Local output/report paths come from CLI options or defaults.

Audit verdict: path strategy prevents collisions between sessions and supports Windows local paths (including spaces).

## 6) Session states and error handling

- Session state transitions are explicit and persisted atomically.
- Ambiguous processing interruption raises dedicated error and stores recovery intent.
- Cleanup warnings are appended without masking primary worker/orchestration errors.
- Corrupted session store blocks new paid workflow.

Audit verdict: state/error behavior matches expected safety principles.

## 7) Concurrency and double-creation protection

- Lock file: `~/.vast_upscaler/workflow.lock` prevents concurrent workflow execution.
- Session ids use UUID-derived suffixes to avoid same-second collisions.

Audit verdict: concurrent paid workflow protection is present.

Limit:

- stale lock cleanup remains operator-managed after abnormal termination.

## 8) Behavior after abrupt interruption

- `KeyboardInterrupt` in workflow transitions session to `PRESERVED_FOR_RECOVERY` and captures recovery flags.
- Resume path is manual-first and attempts artifact recovery before any notion of restart.

Audit verdict: interruption path is conservative and recovery-oriented.

## 9) Destruction verification effectiveness

- Destruction does not trust API success alone.
- Implementation polls `list_instances` until target id disappears or timeout.
- On timeout, session transitions to `CLEANUP_REQUIRED` with explicit warning.

Audit verdict: disappearance verification is implemented.

## 10) Secrets exposure review

- No explicit API key/private key logging found in CLI/orchestrator/transport messages.
- Session report stores operational metadata but no secret material by design.

Audit verdict: no direct secret leak path found in audited modules.

Residual caution:

- external dependencies (libraries/tooling/terminal history) can still leak secrets if misconfigured.

## 11) SSH bootstrap on ephemeral instances

- Strict key checking is enforced.
- Workflow requires operator-assisted trust bootstrap when host is missing/untrusted.

Audit verdict: security behavior is correct for first version.

Not fully validated without real test:

- real host fingerprint retrieval and verification flow under live Vast networking
- known_hosts behavior with hashed host entries and provider host key rotation

## 12) Items requiring real-world validation

- Live Vast API acceptance for chosen offer/image/runtype combinations.
- Actual instance readiness timing variance and network reliability.
- End-to-end transfer throughput/cost impact in real billing context.
- Effective destruction timing and control-plane eventual consistency in production.
- Actual billed cost versus estimate on a real run.

## 13) Audit conclusion

- Code integration A-E is coherent and safety-oriented under simulated execution.
- Real-environment uncertainty remains on provider/network/billing behavior and must be validated via controlled low-cost runbook.