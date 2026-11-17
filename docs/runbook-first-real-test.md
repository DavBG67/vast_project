# Runbook - First Real Orchestrator Test (Windows)

Date: 2026-11-17

Purpose: execute one low-cost controlled real run after approval.

Important:

- Do not run this runbook without explicit go-ahead.
- Keep runtime short and clip length ~5 seconds.
- This runbook assumes phase 4E code is already merged.

## 0) Preconditions

- Windows PowerShell.
- Python environment installed with project dependencies.
- `ffmpeg`, `ffprobe`, `ssh`, `scp`, `ssh-keyscan` available in PATH.
- Vast API key and SSH private key available locally.
- Worker image already published and accessible from Vast instance.

## 1) Prepare environment

```powershell
Set-Location J:\Xampp2\htdocs\vast_project
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
pip install -e ".[dev]"
```

Set secrets (current shell only):

```powershell
$env:VAST_API_KEY = "<your_api_key>"
$env:VAST_SSH_PRIVATE_KEY = "C:\Users\<you>\.ssh\vast_ed25519"
```

## 2) Verify account state before spending

1. Check Vast wallet/credit in Vast console.
2. Confirm there is enough balance for one short test.
3. List currently active instances:

```powershell
upscaler instances
```

If unexpected active instances are present:

- stop and reconcile first
- do not start a new paid run until clear

## 3) Prepare test clip and config

Use a local 5-second MP4 clip, for example:

- `D:\Videos\vast-test-5s.mp4`

Edit `config.yaml` for the first test target:

```yaml
video:
  target_width: 1280
  target_height: 720
  default_profile: balanced

vast:
  gpu_models:
    - RTX 3060
    - RTX 3060 Ti
    - RTX A4000
  max_hourly_price_usd: 0.25
  max_session_hours: 1
  verified_only: true
  min_reliability: 0.98
  disk_gb: 40
```

Note:

- price ceilings and max session hours reduce risk but are not a hard billing guarantee.

## 4) Dry-run first (mandatory)

```powershell
upscaler upscale "D:\Videos\vast-test-5s.mp4" ^
  --worker-image "ghcr.io/<owner>/vast-video-upscaler-worker:dev-sha-<sha>" ^
  --dry-run ^
  --config "config.yaml" ^
  --known-hosts "$HOME\.ssh\known_hosts"
```

Expected:

- no creation/destruction calls
- summary with estimated cost

If dry-run fails:

- stop and fix configuration before any real run

## 5) Real run

```powershell
upscaler upscale "D:\Videos\vast-test-5s.mp4" ^
  --worker-image "ghcr.io/<owner>/vast-video-upscaler-worker:dev-sha-<sha>" ^
  --config "config.yaml" ^
  --known-hosts "$HOME\.ssh\known_hosts"
```

When prompted for paid creation confirmation:

- confirm only if the displayed estimate and constraints are acceptable

If prompted for SSH trust bootstrap:

1. Retrieve host key:

```powershell
ssh-keyscan -p <port> <host>
```

2. Verify fingerprint out-of-band (Vast console/provider context).
3. Append validated key to `known_hosts`.
4. Confirm prompt to continue.

## 6) Validate produced artifacts

Default output paths:

- video: `<input-dir>/<input-stem>-upscaled.<ext>`
- report: `<video-output>.report.json`

Validate report status:

```powershell
Get-Content "D:\Videos\vast-test-5s-upscaled.mp4.report.json"
```

Validate output media quickly:

```powershell
ffprobe -v error -show_format -show_streams -print_format json "D:\Videos\vast-test-5s-upscaled.mp4"
```

Check:

- status=completed in report
- positive frames_processed
- 1280x720 output resolution

## 7) Verify destruction and orphan handling

After workflow completion:

```powershell
upscaler instances
```

If instance remains active or cleanup warning appears:

1. Try explicit destroy:

```powershell
upscaler destroy <instance_id> --yes
```

2. Then run session cleanup:

```powershell
upscaler session cleanup
```

3. Re-check active instances:

```powershell
upscaler instances
```

If still active:

- destroy manually in Vast console
- keep local diagnostic artifacts and session history

## 8) Verify real cost

After run completion and confirmed destruction:

1. Check Vast billing/session details in console.
2. Compare billed amount with local estimate.
3. Record variance and conditions (startup delay, transfer time, shutdown lag).

## 9) Failure playbook

If local process crashes or network drops:

```powershell
upscaler session resume
upscaler instances
upscaler session cleanup
```

Rules:

- do not auto-restart ambiguous processing
- prefer artifact recovery first
- verify remote state before any new run

If session store is corrupted:

- stop paid workflows
- reconcile active instances directly in Vast console
- repair/backup the store before retry
