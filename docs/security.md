# Security and Cost Controls (Phase 4E)

## Secrets

- Keep API keys and SSH private keys out of Git.
- Use environment variables:
  - `VAST_API_KEY`
  - `VAST_SSH_PRIVATE_KEY`
- `config.yaml` is ignored by default.

## Financial safety

Implemented controls in phase 4E:

- Read-only and write clients are separated (`vast_client` vs `vast_write_client`).
- Paid creation requires explicit confirmation in `upscaler upscale` real mode.
- `--dry-run` never creates or destroys instances.
- Offer selection enforces configured price/reliability/disk constraints.
- Session records persist estimated costs and variable-cost notes.
- Local max session duration is enforced as a guardrail.
- Automatic cleanup targets only session-owned instances.
- Destruction verification requires disappearance polling; API success alone is insufficient.

Important limitation:

- Price ceilings and local timeouts reduce risk but do not guarantee final billed cost.

## File safety

- Source local videos are never deleted by orchestration code.
- Download publication is atomic.
- Local overwrite is refused by default unless explicitly enabled.
- Session diagnostics are preserved for recovery on ambiguous failures.

## SSH trust policy (phase 4C+)

Current transport policy:

- Host key validation is mandatory (`StrictHostKeyChecking=yes`).
- A dedicated known_hosts file is required before any SSH/SCP call.
- No plaintext transfer channel is allowed.
- Download publication remains atomic and overwrite is refused unless explicitly requested.

Trust bootstrap strategy for a newly created Vast.ai instance:

1. Acquire SSH endpoint details from Vast API and verify expected provider/offer metadata.
2. Retrieve host keys with `ssh-keyscan -p <port> <host>` from a trusted operator workstation.
3. Compare fingerprint out-of-band with the instance details visible in Vast console/provider context.
4. Append only validated keys to the dedicated known_hosts file used by the session.
5. Keep host key checking strict for all subsequent SSH/SCP operations.

Operator interaction in phase 4E CLI (`upscaler upscale` / `upscaler session resume`):

1. CLI detects missing/untrusted host key entry for `<host>:<port>`.
2. CLI pauses and asks explicit operator confirmation.
3. Operator validates fingerprint and updates the dedicated `known_hosts` file.
4. Operator confirms only after validation.
5. CLI proceeds with strict host-key verification still enabled.

If host identity changes unexpectedly:

- Abort automation immediately.
- Preserve diagnostics.
- Require explicit operator re-validation before any reconnection.

## Estimation safety

- `upscaler estimate` uses user-provided processing speed when available.
- Without benchmark data, estimates are labeled as simulation.
- Unknown costs remain unknown and are not silently fabricated.
- Storage is not double-counted when already included in hourly total pricing.
- Transfer costs can still vary in real billing and are treated as variable/uncertain.

## Crash and orphan recovery

- Local interruption transitions session to `PRESERVED_FOR_RECOVERY`.
- `upscaler session resume` performs manual-first recovery checks.
- `upscaler session cleanup` attempts cleanup only for owned instances.
- If destroy verification fails, session remains in cleanup-required state and operator action is required.

Manual orphan procedure:

1. Run `upscaler instances`.
2. Run `upscaler destroy <instance_id> --yes` for the orphan.
3. Run `upscaler session cleanup`.
4. Re-check with `upscaler instances`.
5. If still active, destroy in Vast console and keep diagnostics.

## Worker runtime safety

- Worker handles SIGINT/SIGTERM and avoids reporting partial success.
- Output is staged to a temporary file and renamed only after validation.
- ffprobe validation checks output existence, resolution, fps tolerance, and duration tolerance.
- CUDA/runtime failures return non-zero exit code and a JSON failure report.
- Phase 3 CI does not run paid GPU workloads.
