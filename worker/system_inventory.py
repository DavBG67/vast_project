from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StaticInventory:
    hardware: dict
    software: dict
    environment: dict
    warnings: list[str]


def collect_static_inventory() -> StaticInventory:
    warnings: list[str] = []

    gpu = _collect_gpu(warnings)
    cpu = _collect_cpu(warnings)
    memory = _collect_memory(warnings)
    storage = _collect_storage(warnings)
    software = _collect_software(warnings)
    environment = _collect_environment(warnings)

    return StaticInventory(
        hardware={
            "gpu": gpu,
            "cpu": cpu,
            "memory": memory,
            "storage": storage,
        },
        software=software,
        environment=environment,
        warnings=warnings,
    )


def _collect_gpu(warnings: list[str]) -> dict:
    rows: list[dict] = []
    try:
        result = _run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,cuda_version",
                "--format=csv,noheader,nounits",
            ]
        )
    except OSError as exc:
        warnings.append(f"gpu_inventory_unavailable: nvidia-smi execution failed ({exc})")
        return {
            "container_visible": {
                "count": None,
                "models": [],
                "total_vram_bytes": None,
                "driver_version": None,
                "cuda_driver_version": None,
            },
            "host_physical": None,
            "provenance": "observed",
        }

    if result.returncode != 0:
        warnings.append("gpu_inventory_unavailable: nvidia-smi returned non-zero exit code")
        return {
            "container_visible": {
                "count": None,
                "models": [],
                "total_vram_bytes": None,
                "driver_version": None,
                "cuda_driver_version": None,
            },
            "host_physical": None,
            "provenance": "observed",
        }

    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        memory_mb = _to_int(parts[1])
        rows.append(
            {
                "model": parts[0] or None,
                "vram_bytes": (memory_mb * 1024 * 1024) if memory_mb is not None else None,
                "driver_version": parts[2] or None,
                "cuda_driver_version": parts[3] or None,
            }
        )

    total_vram = sum(item["vram_bytes"] or 0 for item in rows) if rows else None
    driver = rows[0]["driver_version"] if rows else None
    cuda_driver = rows[0]["cuda_driver_version"] if rows else None
    return {
        "container_visible": {
            "count": len(rows),
            "models": [item["model"] for item in rows if item["model"]],
            "total_vram_bytes": total_vram,
            "driver_version": driver,
            "cuda_driver_version": cuda_driver,
            "per_gpu": [{"model": item["model"], "vram_bytes": item["vram_bytes"]} for item in rows],
        },
        "host_physical": None,
        "provenance": "observed",
    }


def _collect_cpu(warnings: list[str]) -> dict:
    model = None
    cpuinfo = _safe_read_text(Path("/proc/cpuinfo"))
    if cpuinfo is None:
        warnings.append("cpu_inventory_partial: /proc/cpuinfo unavailable")
    else:
        for line in cpuinfo.splitlines():
            if line.lower().startswith("model name"):
                _, _, value = line.partition(":")
                model = value.strip() or None
                break

    quota = _read_cpu_quota_cores(warnings)
    return {
        "container_visible": {
            "model": model,
            "architecture": platform.machine() or None,
            "logical_cpu_count": os.cpu_count(),
            "cgroup_quota_cores": quota,
        },
        "host_physical": None,
        "provenance": "observed",
    }


def _collect_memory(warnings: list[str]) -> dict:
    total_bytes = None
    available_bytes = None
    meminfo = _safe_read_text(Path("/proc/meminfo"))
    if meminfo is None:
        warnings.append("memory_inventory_partial: /proc/meminfo unavailable")
    else:
        mapping: dict[str, int] = {}
        for line in meminfo.splitlines():
            left, _, right = line.partition(":")
            if not right:
                continue
            value = right.strip().split(" ", 1)[0]
            parsed = _to_int(value)
            if parsed is not None:
                mapping[left.strip()] = parsed * 1024
        total_bytes = mapping.get("MemTotal")
        available_bytes = mapping.get("MemAvailable")

    return {
        "container_visible": {
            "total_bytes": total_bytes,
            "available_bytes": available_bytes,
            "cgroup_limit_bytes": _read_memory_limit_bytes(warnings),
        },
        "host_physical": None,
        "provenance": "observed",
    }


def _collect_storage(warnings: list[str]) -> dict:
    total_bytes = None
    available_bytes = None
    try:
        usage = shutil.disk_usage("/")
        total_bytes = usage.total
        available_bytes = usage.free
    except OSError as exc:
        warnings.append(f"storage_inventory_partial: disk usage unavailable ({exc})")

    return {
        "container_visible": {
            "root_mount_total_bytes": total_bytes,
            "root_mount_available_bytes": available_bytes,
            "root_filesystem": _root_filesystem_type(warnings),
        },
        "host_physical": None,
        "provenance": "observed",
    }


def _collect_software(warnings: list[str]) -> dict:
    ffmpeg_version = None
    try:
        ffmpeg = _run_command(["ffmpeg", "-version"])
        if ffmpeg.returncode == 0:
            first = ffmpeg.stdout.splitlines()[0] if ffmpeg.stdout else ""
            ffmpeg_version = first.removeprefix("ffmpeg version ").strip() or None
        else:
            warnings.append("software_inventory_partial: ffmpeg -version failed")
    except OSError as exc:
        warnings.append(f"software_inventory_partial: ffmpeg unavailable ({exc})")

    os_release = _parse_os_release()
    if os_release is None:
        warnings.append("software_inventory_partial: /etc/os-release unavailable")

    torch_version = None
    torch_cuda = None
    try:
        import torch  # type: ignore

        torch_version = str(getattr(torch, "__version__", "")) or None
        torch_cuda = str(getattr(torch.version, "cuda", "")) or None
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"software_inventory_partial: torch metadata unavailable ({exc})")

    return {
        "os": {
            "name": os_release.get("PRETTY_NAME") if os_release else None,
            "kernel": platform.release() or None,
            "architecture": platform.machine() or None,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "pytorch_version": torch_version,
            "pytorch_compiled_cuda_version": torch_cuda,
            "ffmpeg_version": ffmpeg_version,
        },
        "provenance": "observed",
    }


def _collect_environment(_warnings: list[str]) -> dict:
    cgroup_version = _detect_cgroup_version()
    cgroup_text = _safe_read_text(Path("/proc/1/cgroup")) or ""
    return {
        "containerized": Path("/.dockerenv").exists() or "docker" in cgroup_text or "containerd" in cgroup_text or "kubepods" in cgroup_text,
        "cgroup_version": cgroup_version,
        "resource_scope": {
            "container_visible_resources": "observed",
            "host_physical_resources": "unavailable_in_container",
        },
        "sensitive_identifiers_included": False,
        "provenance": "observed",
    }


def _read_cpu_quota_cores(warnings: list[str]) -> float | None:
    cpu_max = _safe_read_text(Path("/sys/fs/cgroup/cpu.max"))
    if cpu_max is not None:
        fields = cpu_max.strip().split()
        if len(fields) >= 2:
            quota, period = fields[0], fields[1]
            if quota == "max":
                return None
            quota_i = _to_int(quota)
            period_i = _to_int(period)
            if quota_i is not None and period_i and period_i > 0:
                return round(quota_i / period_i, 4)
        warnings.append("cpu_inventory_partial: unable to parse cgroup v2 cpu.max")
        return None

    quota_text = _safe_read_text(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
    period_text = _safe_read_text(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
    quota_i = _to_int((quota_text or "").strip())
    period_i = _to_int((period_text or "").strip())
    if quota_i is None or period_i is None:
        return None
    if quota_i < 0:
        return None
    if period_i <= 0:
        warnings.append("cpu_inventory_partial: invalid cgroup v1 cpu period")
        return None
    return round(quota_i / period_i, 4)


def _read_memory_limit_bytes(warnings: list[str]) -> int | None:
    mem_max = _safe_read_text(Path("/sys/fs/cgroup/memory.max"))
    if mem_max is not None:
        value = mem_max.strip()
        if value == "max":
            return None
        parsed = _to_int(value)
        if parsed is None:
            warnings.append("memory_inventory_partial: unable to parse cgroup v2 memory.max")
        return parsed

    value = _safe_read_text(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    if value is None:
        return None
    parsed = _to_int(value.strip())
    if parsed is None:
        warnings.append("memory_inventory_partial: unable to parse cgroup v1 memory limit")
    return parsed


def _root_filesystem_type(warnings: list[str]) -> str | None:
    mounts = _safe_read_text(Path("/proc/mounts"))
    if mounts is None:
        warnings.append("storage_inventory_partial: /proc/mounts unavailable")
        return None
    for line in mounts.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "/":
            return parts[2]
    warnings.append("storage_inventory_partial: root filesystem not found in /proc/mounts")
    return None


def _parse_os_release() -> dict[str, str] | None:
    content = _safe_read_text(Path("/etc/os-release"))
    if content is None:
        return None
    values: dict[str, str] = {}
    for raw in content.splitlines():
        if "=" not in raw or raw.startswith("#"):
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = value.strip().strip('"')
    return values


def _detect_cgroup_version() -> str | None:
    if Path("/sys/fs/cgroup/cgroup.controllers").exists():
        return "v2"
    if Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").exists():
        return "v1"
    return None


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _safe_read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
