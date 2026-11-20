from __future__ import annotations

import shutil
import subprocess
import threading
import time
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .system_inventory import _read_cpu_quota_cores, _read_memory_limit_bytes


SampleProvider = Callable[[list[str], "_CpuSnapshot | None"], tuple[dict[str, Any], "_CpuSnapshot | None"]]


@dataclass(frozen=True)
class TelemetrySnapshot:
    elapsed_seconds: float
    sample: dict[str, Any]


@dataclass(frozen=True)
class TelemetrySummary:
    sampling_interval_seconds: float
    raw_samples: list[dict[str, Any]]
    aggregates: dict[str, Any]
    dropped_samples: int
    max_raw_samples: int
    warnings: list[str]


@dataclass(frozen=True)
class _CpuSnapshot:
    total: int
    idle: int


class TelemetryCollector:
    def __init__(
        self,
        *,
        sampling_interval_seconds: float = 5.0,
        max_raw_samples: int = 10_000,
        sample_provider: SampleProvider | None = None,
    ) -> None:
        self._sampling_interval_seconds = max(0.2, sampling_interval_seconds)
        self._max_raw_samples = max(1, max_raw_samples)
        self._sample_provider = sample_provider or _collect_single_sample
        self._raw_samples: list[TelemetrySnapshot] = []
        self._warnings: list[str] = []
        self._dropped_samples = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._cpu_previous: _CpuSnapshot | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="worker-telemetry", daemon=True)
        self._thread.start()

    def stop(self, *, join_timeout_seconds: float = 3.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=join_timeout_seconds)
            if thread.is_alive():
                with self._lock:
                    self._warnings.append("telemetry_thread_stop_timeout: telemetry thread did not stop before timeout")

    def summary(self) -> TelemetrySummary:
        with self._lock:
            raw = [self._serialize_snapshot(item) for item in self._raw_samples]
            warnings = list(self._warnings)
            dropped = self._dropped_samples
        aggregates = _aggregate_samples(raw)
        return TelemetrySummary(
            sampling_interval_seconds=self._sampling_interval_seconds,
            raw_samples=raw,
            aggregates=aggregates,
            dropped_samples=dropped,
            max_raw_samples=self._max_raw_samples,
            warnings=warnings,
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            started = time.perf_counter()
            warnings: list[str] = []
            try:
                sample, self._cpu_previous = self._sample_provider(warnings, self._cpu_previous)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._warnings.append(f"telemetry_sample_error: {exc}")
                if self._stop_event.wait(self._sampling_interval_seconds):
                    break
                continue

            elapsed = max(0.0, started - (self._started_at or started))
            snapshot = TelemetrySnapshot(elapsed_seconds=elapsed, sample=sample)
            with self._lock:
                self._warnings.extend(warnings)
                self._raw_samples.append(snapshot)
                if len(self._raw_samples) > self._max_raw_samples:
                    self._raw_samples.pop(0)
                    self._dropped_samples += 1
            spent = max(0.0, time.perf_counter() - started)
            remaining = max(0.0, self._sampling_interval_seconds - spent)
            if self._stop_event.wait(remaining):
                break

    @staticmethod
    def _serialize_snapshot(snapshot: TelemetrySnapshot) -> dict[str, Any]:
        missing = _collect_missing_metrics(snapshot.sample)
        payload = {
            "elapsed_seconds": round(snapshot.elapsed_seconds, 3),
            **snapshot.sample,
        }
        if missing:
            payload["missing_metrics"] = missing
        return payload


def _collect_single_sample(
    warnings: list[str],
    previous_cpu: _CpuSnapshot | None,
) -> tuple[dict[str, Any], _CpuSnapshot | None]:
    cpu_quota = _read_cpu_quota_cores(warnings)
    sample_cpu, next_cpu = _collect_cpu_sample(previous_cpu, cpu_quota)
    sample_memory = _collect_memory_sample(warnings)
    sample_storage = _collect_storage_sample(warnings)
    sample_gpu = _collect_gpu_sample(warnings)

    return (
        {
            "provenance": "observed",
            "cpu": sample_cpu,
            "memory": sample_memory,
            "storage": sample_storage,
            "gpu": sample_gpu,
        },
        next_cpu,
    )


def _collect_gpu_sample(warnings: list[str]) -> dict[str, Any]:
    query_fields = [
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "temperature.gpu",
        "power.draw",
        "clocks.sm",
        "clocks.mem",
        "clocks_throttle_reasons.active",
        "clocks_throttle_reasons.hw_thermal_slowdown",
        "clocks_throttle_reasons.hw_power_brake_slowdown",
    ]
    query = ",".join(query_fields)
    try:
        run = _run_command(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            timeout_seconds=1.5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        warnings.append(f"telemetry_gpu_unavailable: nvidia-smi execution failed ({exc})")
        return {"per_gpu": [], "available": False}

    if run.returncode != 0:
        warnings.append("telemetry_gpu_unavailable: nvidia-smi returned non-zero exit code")
        return {"per_gpu": [], "available": False}

    per_gpu: list[dict[str, Any]] = []
    for row in run.stdout.splitlines():
        row = row.strip()
        if not row:
            continue
        parts = [item.strip() for item in row.split(",")]
        while len(parts) < len(query_fields):
            parts.append("")
        per_gpu.append(
            {
                "utilization_gpu_percent": _to_float(parts[0]),
                "utilization_memory_percent": _to_float(parts[1]),
                "memory_used_bytes": _mb_to_bytes(parts[2]),
                "temperature_celsius": _to_float(parts[3]),
                "power_watts": _to_float(parts[4]),
                "graphics_clock_mhz": _to_float(parts[5]),
                "memory_clock_mhz": _to_float(parts[6]),
                "throttle_active": _parse_bool(parts[7]),
                "thermal_slowdown": _parse_bool(parts[8]),
                "power_brake_slowdown": _parse_bool(parts[9]),
            }
        )

    return {"available": True, "per_gpu": per_gpu}


def _collect_cpu_sample(previous: _CpuSnapshot | None, cpu_quota_cores: float | None) -> tuple[dict[str, Any], _CpuSnapshot | None]:
    current = _read_proc_stat_cpu_total()
    usage_percent = None
    usage_percent_of_quota = None
    if previous is not None and current is not None:
        total_delta = current.total - previous.total
        idle_delta = current.idle - previous.idle
        if total_delta > 0:
            usage_percent = max(0.0, min(100.0, (1.0 - (idle_delta / total_delta)) * 100.0))
            if cpu_quota_cores is not None and cpu_quota_cores > 0:
                logical = _logical_cpu_count()
                if logical and logical > 0:
                    quota_capacity_percent = (cpu_quota_cores / logical) * 100.0
                    if quota_capacity_percent > 0:
                        usage_percent_of_quota = (usage_percent / quota_capacity_percent) * 100.0

    next_state = current or previous
    return (
        {
            "logical_cpu_count": _logical_cpu_count(),
            "usage_percent": _rounded_or_none(usage_percent),
            "usage_percent_of_cgroup_quota": _rounded_or_none(usage_percent_of_quota),
            "cgroup_quota_cores": cpu_quota_cores,
        },
        next_state,
    )


def _collect_memory_sample(warnings: list[str]) -> dict[str, Any]:
    total_bytes = None
    available_bytes = None
    meminfo = _safe_read_text(Path("/proc/meminfo"))
    if meminfo is not None:
        values: dict[str, int] = {}
        for line in meminfo.splitlines():
            left, _, right = line.partition(":")
            if not right:
                continue
            parsed = _to_int(right.strip().split(" ", 1)[0])
            if parsed is not None:
                values[left.strip()] = parsed * 1024
        total_bytes = values.get("MemTotal")
        available_bytes = values.get("MemAvailable")
    else:
        warnings.append("telemetry_memory_partial: /proc/meminfo unavailable")

    used_bytes = None
    if total_bytes is not None and available_bytes is not None:
        used_bytes = max(0, total_bytes - available_bytes)

    cgroup_limit = _read_memory_limit_bytes(warnings)
    cgroup_usage = _read_cgroup_memory_usage_bytes()
    return {
        "total_bytes": total_bytes,
        "available_bytes": available_bytes,
        "used_bytes": used_bytes,
        "cgroup_limit_bytes": cgroup_limit,
        "cgroup_usage_bytes": cgroup_usage,
    }


def _collect_storage_sample(warnings: list[str]) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage("/")
        return {
            "root_available_bytes": usage.free,
            "root_total_bytes": usage.total,
        }
    except OSError as exc:
        warnings.append(f"telemetry_storage_partial: disk usage unavailable ({exc})")
        return {"root_available_bytes": None, "root_total_bytes": None}


def _aggregate_samples(raw_samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not raw_samples:
        return {}
    stripped = [dict(item) for item in raw_samples]
    for item in stripped:
        item.pop("elapsed_seconds", None)
        item.pop("missing_metrics", None)
    return _aggregate_node(stripped)


def _aggregate_node(node: Any) -> Any:
    if isinstance(node, list):
        if not node:
            return {}
        if all(isinstance(item, dict) for item in node):
            keys: set[str] = set()
            for item in node:
                keys.update(item.keys())
            output: dict[str, Any] = {}
            for key in sorted(keys):
                values = [item.get(key) for item in node]
                aggregated = _aggregate_node(values)
                if aggregated not in ({}, None):
                    output[key] = aggregated
            return output
        numeric_values = [float(item) for item in node if isinstance(item, (int, float)) and not isinstance(item, bool)]
        if numeric_values:
            count = len(numeric_values)
            return {
                "min": min(numeric_values),
                "avg": sum(numeric_values) / count,
                "max": max(numeric_values),
                "count": count,
            }
        if all(item is None for item in node):
            return None
        if all(isinstance(item, list) for item in node):
            max_len = max(len(item) for item in node)
            return [_aggregate_node([item[index] if index < len(item) else None for item in node]) for index in range(max_len)]
        return {}
    return None


def _collect_missing_metrics(sample: dict[str, Any]) -> list[str]:
    missing: list[str] = []

    def _walk(obj: Any, prefix: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                next_prefix = f"{prefix}.{key}" if prefix else key
                _walk(value, next_prefix)
            return
        if isinstance(obj, list):
            for index, value in enumerate(obj):
                _walk(value, f"{prefix}[{index}]")
            return
        if obj is None:
            missing.append(prefix)

    _walk(sample, "")
    return missing


def _read_proc_stat_cpu_total() -> _CpuSnapshot | None:
    content = _safe_read_text(Path("/proc/stat"))
    if content is None:
        return None
    for line in content.splitlines():
        if not line.startswith("cpu "):
            continue
        parts = line.split()
        if len(parts) < 5:
            return None
        values = [int(value) for value in parts[1:] if value.isdigit()]
        if len(values) < 5:
            return None
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)
        return _CpuSnapshot(total=total, idle=idle)
    return None


def _read_cgroup_memory_usage_bytes() -> int | None:
    value = _safe_read_text(Path("/sys/fs/cgroup/memory.current"))
    if value is not None:
        return _to_int(value.strip())
    value = _safe_read_text(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
    if value is not None:
        return _to_int(value.strip())
    return None


def _logical_cpu_count() -> int | None:
    count = os.cpu_count()
    return int(count) if count is not None else None


def _safe_read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _run_command(command: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout_seconds)


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized or normalized in {"[not supported]", "n/a", "na"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def _mb_to_bytes(value: str | None) -> int | None:
    as_float = _to_float(value)
    if as_float is None:
        return None
    return int(as_float * 1024 * 1024)


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"active", "enabled", "true", "1"}:
        return True
    if normalized in {"not active", "disabled", "false", "0"}:
        return False
    return None


def _rounded_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)
