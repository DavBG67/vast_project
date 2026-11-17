# Security and Cost Controls (Phase 3)

## Secrets

- Keep API keys and SSH private keys out of Git.
- Use environment variables:
  - `VAST_API_KEY`
  - `VAST_SSH_PRIVATE_KEY`
- `config.yaml` is ignored by default.

## Financial safety

Phase 2 still does not perform paid operations.

Implemented controls:

- Read-only endpoint allowlist in `vast_client`.
- Runtime block on non-whitelisted endpoints.
- No create/start/stop/destroy instance method in CLI.
- Explicit offline mode for tests and demos.
- Retry with bounded backoff for transient errors only.
- Worker image contains no API keys or private SSH keys.
- Worker runs as non-root user in container.

Future phases must enforce:

- Explicit confirmation before any rental.
- Session budget ceiling.
- Time limit and watchdog strategy.
- Verified instance destruction by API response.

## File safety

- Source local videos are never deleted by phase 1 code.
- Existing output overwrite logic is not implemented yet and must be explicit in phase 4.

## SSH policy (future)

When transfers are implemented:

- Host key validation must remain enabled.
- No plaintext transfer channels.
- Transfer completion requires size/checksum verification.

## Estimation safety

- `upscaler estimate` uses user-provided processing speed when available.
- Without benchmark data, estimates are labeled as simulation.
- Unknown costs remain unknown and are not silently fabricated.
- Storage is not double-counted when already included in hourly total pricing.

## Worker runtime safety

- Worker handles SIGINT/SIGTERM and avoids reporting partial success.
- Output is staged to a temporary file and renamed only after validation.
- ffprobe validation checks output existence, resolution, fps tolerance, and duration tolerance.
- CUDA/runtime failures return non-zero exit code and a JSON failure report.
- Phase 3 CI does not run paid GPU workloads.
