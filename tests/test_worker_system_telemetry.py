from __future__ import annotations

import subprocess
import time

from worker.system_telemetry import TelemetryCollector, _collect_single_sample, _collect_gpu_sample


def test_telemetry_collector_collects_and_aggregates_samples() -> None:
    values = iter(
        [
            ({"cpu": {"usage_percent": 10.0}, "memory": {"used_bytes": 100}, "gpu": {"per_gpu": []}}, None),
            ({"cpu": {"usage_percent": 30.0}, "memory": {"used_bytes": 300}, "gpu": {"per_gpu": []}}, None),
            ({"cpu": {"usage_percent": 20.0}, "memory": {"used_bytes": 200}, "gpu": {"per_gpu": []}}, None),
        ]
    )

    def _provider(_warnings, prev):
        _ = prev
        try:
            return next(values)
        except StopIteration:
            return ({"cpu": {"usage_percent": 20.0}, "memory": {"used_bytes": 200}, "gpu": {"per_gpu": []}}, None)

    collector = TelemetryCollector(sampling_interval_seconds=0.2, max_raw_samples=10, sample_provider=_provider)
    collector.start()
    time.sleep(0.55)
    collector.stop()
    summary = collector.summary()

    assert len(summary.raw_samples) >= 2
    assert summary.aggregates["cpu"]["usage_percent"]["count"] >= 2
    assert summary.aggregates["memory"]["used_bytes"]["max"] >= summary.aggregates["memory"]["used_bytes"]["min"]


def test_telemetry_collector_handles_sampling_exception() -> None:
    state = {"calls": 0}

    def _provider(_warnings, prev):
        _ = prev
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("boom")
        return ({"cpu": {"usage_percent": 50.0}}, None)

    collector = TelemetryCollector(sampling_interval_seconds=0.2, max_raw_samples=5, sample_provider=_provider)
    collector.start()
    time.sleep(0.45)
    collector.stop()
    summary = collector.summary()

    assert any("telemetry_sample_error" in warning for warning in summary.warnings)
    assert len(summary.raw_samples) >= 1


def test_telemetry_collector_limits_raw_sample_memory() -> None:
    def _provider(_warnings, prev):
        _ = prev
        return ({"cpu": {"usage_percent": 10.0}}, None)

    collector = TelemetryCollector(sampling_interval_seconds=0.2, max_raw_samples=2, sample_provider=_provider)
    collector.start()
    time.sleep(0.7)
    collector.stop()
    summary = collector.summary()

    assert len(summary.raw_samples) <= 2
    assert summary.dropped_samples >= 1


def test_collect_gpu_sample_handles_missing_nvidia_smi(monkeypatch) -> None:
    def _run_command(_command, *, timeout_seconds):
        _ = timeout_seconds
        raise OSError("not found")

    monkeypatch.setattr("worker.system_telemetry._run_command", _run_command)
    warnings: list[str] = []
    sample = _collect_gpu_sample(warnings)
    assert sample["available"] is False
    assert any("telemetry_gpu_unavailable" in warning for warning in warnings)


def test_collect_gpu_sample_supports_multiple_gpus(monkeypatch) -> None:
    def _run_command(_command, *, timeout_seconds):
        _ = timeout_seconds
        return subprocess.CompletedProcess(
            _command,
            0,
            stdout=(
                "50, 25, 1024, 60, 120, 1800, 7000, Active, Not Active, Not Active\n"
                "20, 10, 512, 55, 80, 1500, 6500, Not Active, Not Active, Not Active\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("worker.system_telemetry._run_command", _run_command)
    warnings: list[str] = []
    sample = _collect_gpu_sample(warnings)
    assert sample["available"] is True
    assert len(sample["per_gpu"]) == 2
    assert sample["per_gpu"][0]["utilization_gpu_percent"] == 50.0
    assert sample["per_gpu"][1]["memory_used_bytes"] == 512 * 1024 * 1024


def test_collect_single_sample_keeps_partial_metrics(monkeypatch) -> None:
    def _run_command(_command, *, timeout_seconds):
        _ = timeout_seconds
        return subprocess.CompletedProcess(_command, 1, stdout="", stderr="err")

    def _safe_read_text(path):
        normalized = str(path).replace("\\", "/")
        mapping = {
            "/proc/stat": "cpu  100 0 100 800 0 0 0 0 0 0\n",
            "/proc/meminfo": "MemTotal: 1000 kB\nMemAvailable: 400 kB\n",
            "/sys/fs/cgroup/cpu.max": "max 100000\n",
            "/sys/fs/cgroup/memory.max": "max\n",
            "/sys/fs/cgroup/memory.current": "123456\n",
        }
        return mapping.get(normalized)

    monkeypatch.setattr("worker.system_telemetry._run_command", _run_command)
    monkeypatch.setattr("worker.system_telemetry._safe_read_text", _safe_read_text)
    warnings: list[str] = []
    sample, state = _collect_single_sample(warnings, None)
    assert state is not None
    assert sample["memory"]["used_bytes"] == (1000 - 400) * 1024
    assert sample["gpu"]["available"] is False
    assert any("telemetry_gpu_unavailable" in warning for warning in warnings)
