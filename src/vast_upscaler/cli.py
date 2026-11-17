from __future__ import annotations

import json
import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import typer
from rich.console import Console
from rich.table import Table

from vast_upscaler import __version__
from vast_upscaler.config import ConfigError, Settings, load_config
from vast_upscaler.cost import estimate_cost
from vast_upscaler.offers import (
    OfferConstraints,
    build_offers_query,
    filter_offers,
    normalize_offers,
    rank_offers,
)
from vast_upscaler.orchestrator import (
    OrchestrationError,
    OrchestrationOptions,
    SessionExecutionLock,
    cleanup_sessions,
    destroy_instance_with_verification,
    ensure_store_readable,
    list_active_instances,
    resume_session,
    run_upscale_workflow,
)
from vast_upscaler.session_store import SessionStore
from vast_upscaler.transfer import OpenSshTransport
from vast_upscaler.vast_client import (
    VastApiError,
    VastAuthError,
    VastClient,
    VastClientConfig,
)
from vast_upscaler.vast_write_client import VastWriteApiError, VastWriteClient
from vast_upscaler.video import VideoProbeError, inspect_video

app = typer.Typer(help="Vast Video Upscaler local orchestrator CLI")
session_app = typer.Typer(help="Session recovery and cleanup commands")
console = Console()

CONFIG_OPTION = typer.Option(Path("config.yaml"), exists=False, dir_okay=False)
OFFLINE_FIXTURE_OPTION = typer.Option(None, "--fixture", help="Offline JSON fixture path.")
SORT_BY_OPTION = typer.Option("price", "--sort-by")
LIMIT_OPTION = typer.Option(10, min=1, max=100)
MAX_PRICE_OPTION = typer.Option(None, "--max-price-usd")
FPS_PROCESSING_OPTION = typer.Option(None, "--fps-processing", min=0.001)
OUTPUT_OPTION = typer.Option(None, "--output")
REPORT_OPTION = typer.Option(None, "--report")
WORKER_IMAGE_OPTION = typer.Option(..., "--worker-image")
DRY_RUN_OPTION = typer.Option(False, "--dry-run")
KEEP_INSTANCE_OPTION = typer.Option(False, "--keep-instance")
KNOWN_HOSTS_OPTION = typer.Option(Path.home() / ".ssh" / "known_hosts", "--known-hosts")
SKIP_CONFIRM_OPTION = typer.Option(False, "--yes", help="Skip confirmation prompt.")
SESSION_ID_OPTION = typer.Option(None, "--session-id")

app.add_typer(session_app, name="session")


@app.command()
def doctor(config: Path = CONFIG_OPTION) -> None:
    """Validate local prerequisites without touching paid Vast.ai operations."""
    checks: list[tuple[str, bool, str]] = []

    python_ok = platform.python_version_tuple() >= ("3", "11", "0")
    checks.append(("python", python_ok, platform.python_version()))

    ffprobe_path = shutil.which("ffprobe")
    checks.append(("ffprobe", ffprobe_path is not None, ffprobe_path or "not found"))

    ffmpeg_path = shutil.which("ffmpeg")
    checks.append(("ffmpeg", ffmpeg_path is not None, ffmpeg_path or "not found"))

    try:
        settings = load_config(config)
        checks.append(("config", True, str(config if config.exists() else "defaults")))
    except ConfigError as exc:
        checks.append(("config", False, str(exc)))
        settings = Settings()

    api_key_present = bool(settings.secrets.vast_api_key)
    checks.append(("VAST_API_KEY", True, "set" if api_key_present else "missing (optional in phase 1)"))

    ssh_key_path = settings.secrets.ssh_private_key_path
    ssh_key_ok = ssh_key_path is None or Path(ssh_key_path).exists()
    checks.append(("ssh_key", ssh_key_ok, ssh_key_path or "not configured"))

    failed = 0
    for name, ok, detail in checks:
        status = "OK" if ok else "FAIL"
        typer.echo(f"{status:4} {name:12} {detail}")
        if not ok:
            failed += 1

    if failed:
        raise typer.Exit(code=1)


@app.command()
def inspect(video_path: Path, json_output: bool = typer.Option(False, "--json")) -> None:
    """Inspect source video metadata using ffprobe."""
    try:
        info = inspect_video(video_path)
    except VideoProbeError as exc:
        typer.echo(f"Inspection failed: {exc}")
        raise typer.Exit(code=1)

    payload: dict[str, Any] = info.model_dump(mode="json")
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    for key, value in payload.items():
        typer.echo(f"{key}: {value}")


@app.command()
def status() -> None:
    """Show the latest local session state."""
    store = SessionStore()
    current = store.current()
    if current is None:
        typer.echo("No recorded session yet.")
        return

    for key, value in current.model_dump(mode="json").items():
        typer.echo(f"{key}: {value}")


@app.command()
def history(limit: int = typer.Option(10, min=1, max=100)) -> None:
    """Show recent local sessions."""
    store = SessionStore()
    sessions = store.list_recent(limit=limit)
    if not sessions:
        typer.echo("No session history yet.")
        return

    for session in sessions:
        typer.echo(
            f"{session.started_at} | id={session.session_id} | state={session.state} "
            f"| instance={session.instance_id or '-'}"
        )


@app.command()
def offers(
    fixture: Path | None = OFFLINE_FIXTURE_OPTION,
    config: Path = CONFIG_OPTION,
    sort_by: Literal["price", "reliability", "performance"] = SORT_BY_OPTION,
    limit: int = LIMIT_OPTION,
    max_price: float | None = MAX_PRICE_OPTION,
) -> None:
    settings = _load_settings_or_exit(config)
    constraints = _build_constraints(settings=settings, max_price_override=max_price)

    raw_offers, source = _load_raw_offers(settings=settings, constraints=constraints, fixture=fixture, limit=100)
    candidates = normalize_offers(raw_offers)
    compatible = filter_offers(candidates, constraints)
    ranked = rank_offers(compatible, sort_by=sort_by)[:limit]

    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    console.print(f"Offers source: {source}")
    console.print(f"Fetched at (UTC): {fetched_at}")
    console.print(f"Ranking mode: {sort_by} (provisional; benchmark data not integrated yet)")

    if not ranked:
        console.print("No compatible offers found with current constraints.")
        raise typer.Exit(code=1)

    table = Table(title="Compatible Vast.ai Offers")
    table.add_column("GPU")
    table.add_column("Country/Region")
    table.add_column("Price/h (USD)", justify="right")
    table.add_column("Reliability", justify="right")
    table.add_column("RAM", justify="right")
    table.add_column("Disk", justify="right")
    table.add_column("ID", justify="right")

    for item in ranked:
        table.add_row(
            item.gpu_name,
            item.country_or_region,
            _fmt_money(item.dph_total),
            _fmt_ratio(item.reliability),
            _fmt_ram_mb(item.cpu_ram_mb),
            _fmt_disk(item.disk_space_gb),
            str(item.offer_id),
        )

    console.print(table)


@app.command()
def benchmark(video_path: Path) -> None:
    _ = video_path
    typer.echo("Phase 2+: benchmark command is not implemented yet.")
    raise typer.Exit(code=2)


@app.command()
def estimate(
    video_path: Path,
    fps_processing: float | None = FPS_PROCESSING_OPTION,
    fixture: Path | None = OFFLINE_FIXTURE_OPTION,
    config: Path = CONFIG_OPTION,
    sort_by: Literal["price", "reliability", "performance"] = SORT_BY_OPTION,
    max_price: float | None = MAX_PRICE_OPTION,
) -> None:
    settings = _load_settings_or_exit(config)
    constraints = _build_constraints(settings=settings, max_price_override=max_price)

    try:
        video = inspect_video(video_path)
    except VideoProbeError as exc:
        console.print(f"Inspection failed: {exc}")
        raise typer.Exit(code=1)

    raw_offers, source = _load_raw_offers(settings=settings, constraints=constraints, fixture=fixture, limit=100)
    candidates = normalize_offers(raw_offers)
    compatible = filter_offers(candidates, constraints)
    ranked = rank_offers(compatible, sort_by=sort_by)
    if not ranked:
        console.print("No compatible offer found for estimation.")
        raise typer.Exit(code=1)

    selected = ranked[0]
    estimate_result = estimate_cost(
        video=video,
        offer=selected,
        processing_fps_assumption=fps_processing,
        allocated_storage_gb=settings.vast.disk_gb,
    )

    summary = Table(title="Cost Estimation (Phase 2)")
    summary.add_column("Field")
    summary.add_column("Value")
    summary.add_row("Source", source)
    summary.add_row("Video duration", _fmt_seconds(estimate_result.duration_seconds))
    summary.add_row("Resolution", estimate_result.resolution)
    summary.add_row("Estimated frames", _fmt_count(estimate_result.estimated_frames))
    summary.add_row("Selected GPU", estimate_result.selected_gpu)
    summary.add_row("Offer ID", str(estimate_result.selected_offer_id))
    summary.add_row("Rental price/h", _fmt_money(estimate_result.rental_total_hourly_usd))
    summary.add_row("GPU price/h", _fmt_money(estimate_result.gpu_hourly_usd))
    summary.add_row("Storage price/h", _fmt_money(estimate_result.storage_hourly_usd))
    summary.add_row("Processing fps assumption", _fmt_fps(estimate_result.processing_fps_assumption))
    summary.add_row("Estimated processing duration", _fmt_hours(estimate_result.processing_hours_estimated))
    summary.add_row("Estimated rental cost", _fmt_money(estimate_result.gpu_cost_estimated_usd))
    summary.add_row("Estimated storage cost", _fmt_money(estimate_result.storage_cost_estimated_usd))
    summary.add_row("Estimated transfer upload", "unknown")
    summary.add_row("Estimated transfer download", "unknown")
    summary.add_row(
        "Estimated total",
        _fmt_money(estimate_result.total_estimated_usd)
        + (" (partial)" if estimate_result.total_is_partial else ""),
    )
    console.print(summary)

    notes = Table(title="Assumptions and Limits")
    notes.add_column("Note")
    for note in estimate_result.notes:
        notes.add_row(note)
    notes.add_row("No paid Vast.ai operation is executed by this command.")
    console.print(notes)


@app.command()
def upscale(
    video_path: Path,
    config: Path = CONFIG_OPTION,
    output_path: Path | None = OUTPUT_OPTION,
    report_path: Path | None = REPORT_OPTION,
    worker_image: str = WORKER_IMAGE_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    keep_instance: bool = KEEP_INSTANCE_OPTION,
    known_hosts: Path = KNOWN_HOSTS_OPTION,
    sort_by: Literal["price", "reliability", "performance"] = SORT_BY_OPTION,
    fps_processing: float | None = FPS_PROCESSING_OPTION,
) -> None:
    settings = _load_settings_or_exit(config)
    store = SessionStore()

    try:
        ensure_store_readable(store)
    except OrchestrationError as exc:
        console.print(f"Safety stop: {exc}")
        raise typer.Exit(code=1) from exc

    out_path = output_path or video_path.with_name(f"{video_path.stem}-upscaled{video_path.suffix or '.mkv'}")
    rep_path = report_path or out_path.with_suffix(out_path.suffix + ".report.json")

    read_client = _build_read_client(settings=settings)
    write_client = _build_write_client(settings=settings) if not dry_run else _NoopWriteClient()
    transport = OpenSshTransport()

    try:
        lock_cm = SessionExecutionLock() if not dry_run else _NoopLock()
        with lock_cm:
            summary = run_upscale_workflow(
                settings=settings,
                options=OrchestrationOptions(
                    input_path=video_path,
                    output_path=out_path,
                    report_path=rep_path,
                    worker_image=worker_image,
                    profile=settings.video.default_profile,
                    target_width=settings.video.target_width,
                    target_height=settings.video.target_height,
                    dry_run=dry_run,
                    keep_instance=keep_instance,
                    known_hosts_path=known_hosts,
                    processing_fps=fps_processing,
                    sort_by=sort_by,
                ),
                session_store=store,
                read_client=read_client,
                write_client=write_client,
                transport=transport,
                confirm_paid=lambda msg: typer.confirm(msg, default=False),
                confirm_ssh_trust=lambda msg: typer.confirm(msg, default=False),
            )
    except (OrchestrationError, VastApiError, VastWriteApiError, VideoProbeError) as exc:
        console.print(f"Upscale failed: {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"Upscale failed: {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        if isinstance(read_client, VastClient):
            read_client.close()
        if isinstance(write_client, VastWriteClient):
            write_client.close()

    _print_upscale_summary(summary, dry_run=dry_run)


@app.command()
def instances(config: Path = CONFIG_OPTION) -> None:
    settings = _load_settings_or_exit(config)
    client = _build_write_client(settings=settings)
    try:
        rows = list_active_instances(client=client)
    except VastWriteApiError as exc:
        console.print(f"Failed to list instances: {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"Failed to list instances: {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        client.close()

    table = Table(title="Active Vast.ai Instances")
    table.add_column("ID", justify="right")
    table.add_column("Label")
    table.add_column("Status")
    table.add_column("State")
    for row in rows:
        table.add_row(
            str(row.get("id", "-")),
            str(row.get("label", "-")),
            str(row.get("actual_status", "-")),
            str(row.get("cur_state", "-")),
        )
    console.print(table)


@app.command()
def destroy(
    instance_id: int,
    config: Path = CONFIG_OPTION,
    yes: bool = SKIP_CONFIRM_OPTION,
) -> None:
    settings = _load_settings_or_exit(config)
    if not yes and not typer.confirm(f"Destroy instance {instance_id}?", default=False):
        raise typer.Exit(code=1)

    client = _build_write_client(settings=settings)
    try:
        ok = destroy_instance_with_verification(client=client, instance_id=instance_id)
    except VastWriteApiError as exc:
        console.print(f"Destroy failed: {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"Destroy failed: {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        client.close()

    if not ok:
        console.print("Destroy requested but disappearance could not be verified.")
        raise typer.Exit(code=1)
    console.print(f"Instance {instance_id} destroyed and disappearance verified.")


@session_app.command("resume")
def session_resume(
    session_id: str | None = SESSION_ID_OPTION,
    config: Path = CONFIG_OPTION,
    known_hosts: Path = KNOWN_HOSTS_OPTION,
) -> None:
    settings = _load_settings_or_exit(config)
    client = _build_write_client(settings=settings)
    store = SessionStore()
    try:
        with SessionExecutionLock():
            summary = resume_session(
                store=store,
                session_id=session_id,
                write_client=client,
                settings=settings,
                transport=OpenSshTransport(),
                known_hosts_path=known_hosts,
                ssh_private_key_path=Path(settings.secrets.ssh_private_key_path)
                if settings.secrets.ssh_private_key_path
                else None,
                confirm_ssh_trust=lambda msg: typer.confirm(msg, default=False),
            )
    except (OrchestrationError, VastWriteApiError) as exc:
        console.print(f"Resume failed: {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"Resume failed: {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        client.close()

    console.print(f"Session {summary.session_id}: {summary.state}")
    console.print(summary.message)


@session_app.command("cleanup")
def session_cleanup(config: Path = CONFIG_OPTION) -> None:
    settings = _load_settings_or_exit(config)
    client = _build_write_client(settings=settings)
    store = SessionStore()
    try:
        with SessionExecutionLock():
            results = cleanup_sessions(store=store, client=client)
    except (OrchestrationError, VastWriteApiError) as exc:
        console.print(f"Cleanup failed: {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"Cleanup failed: {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        client.close()

    if not results:
        console.print("No cleanup-eligible sessions.")
        return

    table = Table(title="Session Cleanup")
    table.add_column("Session")
    table.add_column("Attempted")
    table.add_column("Cleaned")
    table.add_column("Warning")
    for item in results:
        table.add_row(item.session_id, str(item.attempted), str(item.cleaned), item.warning or "-")
    console.print(table)


@app.command()
def version() -> None:
    typer.echo(__version__)


def run() -> None:
    app()


def _load_settings_or_exit(config: Path) -> Settings:
    try:
        return load_config(config)
    except ConfigError as exc:
        console.print(f"Invalid config: {exc}")
        raise typer.Exit(code=1) from exc


def _build_constraints(settings: Settings, max_price_override: float | None) -> OfferConstraints:
    max_price = max_price_override if max_price_override is not None else settings.vast.max_hourly_price_usd
    return OfferConstraints(
        gpu_models=settings.vast.gpu_models,
        max_hourly_price_usd=max_price,
        min_reliability=settings.vast.min_reliability,
        verified_only=settings.vast.verified_only,
        disk_gb=settings.vast.disk_gb,
    )


def _load_raw_offers(
    settings: Settings,
    constraints: OfferConstraints,
    fixture: Path | None,
    limit: int,
) -> tuple[list[dict[str, Any]], str]:
    if fixture is not None:
        if not fixture.exists():
            console.print(f"Fixture not found: {fixture}")
            raise typer.Exit(code=1)
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        raw = payload.get("offers") if isinstance(payload, dict) else payload
        if not isinstance(raw, list):
            console.print("Fixture format invalid: expected a JSON list or object with 'offers' list.")
            raise typer.Exit(code=1)
        return raw, f"offline fixture ({fixture})"

    api_key = settings.secrets.vast_api_key
    if not api_key:
        console.print("VAST_API_KEY missing. Use --fixture for offline mode or set the environment variable.")
        raise typer.Exit(code=1)

    query = build_offers_query(constraints=constraints, limit=limit)
    client_config = VastClientConfig()
    try:
        with VastClient(api_key=api_key, config=client_config) as client:
            return client.search_offers(query), "live Vast.ai read-only API"
    except VastAuthError as exc:
        console.print(f"Authentication error: {exc}")
        raise typer.Exit(code=1) from exc
    except VastApiError as exc:
        console.print(f"Vast API error: {exc}")
        raise typer.Exit(code=1) from exc


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.3f}"


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value * 100.0:.2f}%"


def _fmt_ram_mb(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / 1024.0:.1f} GB"


def _fmt_disk(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.0f} GB"


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.2f} s"


def _fmt_count(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.0f}"


def _fmt_fps(value: float | None) -> str:
    if value is None:
        return "not provided"
    return f"{value:.3f} fps"


def _fmt_hours(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.3f} h"


def _build_read_client(settings: Settings):
    api_key = settings.secrets.vast_api_key
    if not api_key:
        raise OrchestrationError("VAST_API_KEY missing.")
    return VastClient(api_key=api_key, config=VastClientConfig())


def _build_write_client(settings: Settings) -> VastWriteClient:
    api_key = settings.secrets.vast_api_key
    if not api_key:
        raise OrchestrationError("VAST_API_KEY missing.")
    return VastWriteClient(api_key=api_key)


def _print_upscale_summary(summary, *, dry_run: bool) -> None:
    mode = "DRY-RUN" if dry_run else "REAL"
    table = Table(title=f"Upscale Summary ({mode})")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Session", summary.session_id)
    table.add_row("State", summary.state)
    table.add_row("Offer ID", str(summary.offer_id))
    table.add_row("GPU", summary.offer_gpu)
    table.add_row("Estimated total", _fmt_money(summary.estimate.total_estimated_usd))
    table.add_row("Max known hourly", _fmt_money(summary.estimate.rental_total_hourly_usd))
    table.add_row("Cleanup verified", str(summary.cleanup_verified))
    table.add_row("Cleanup warning", summary.cleanup_warning or "-")
    if summary.worker_result is not None:
        table.add_row("Output", str(summary.worker_result.local_output_path))
        table.add_row("Report", str(summary.worker_result.local_report_path))
        table.add_row("Frames processed", str(summary.worker_result.frames_processed))
        table.add_row("Duration", _fmt_seconds(summary.worker_result.video_info.duration_seconds))
    console.print(table)
    console.print("Reminder: local timeout is not a guaranteed billing ceiling.")


class _NoopWriteClient:
    def close(self) -> None:
        return


class _NoopLock:
    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        return
