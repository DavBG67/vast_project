from __future__ import annotations

import subprocess

from worker.system_inventory import collect_static_inventory


def test_collect_static_inventory_handles_missing_nvidia_smi(monkeypatch) -> None:
    def _run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "nvidia-smi":
            raise OSError("not found")
        if cmd[0] == "ffmpeg":
            return subprocess.CompletedProcess(cmd, 0, stdout="ffmpeg version 6.1\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def _safe_read_text(path):
        normalized = str(path).replace("\\", "/")
        mapping = {
            "/proc/cpuinfo": "model name\t: Test CPU\n",
            "/proc/meminfo": "MemTotal: 1000 kB\nMemAvailable: 500 kB\n",
            "/proc/mounts": "overlay / overlay rw 0 0\n",
            "/etc/os-release": 'PRETTY_NAME="TestOS"\n',
            "/proc/1/cgroup": "0::/\n",
            "/sys/fs/cgroup/cpu.max": "max 100000\n",
            "/sys/fs/cgroup/memory.max": "max\n",
        }
        return mapping.get(normalized)

    monkeypatch.setattr("worker.system_inventory._run_command", _run_command)
    monkeypatch.setattr("worker.system_inventory._safe_read_text", _safe_read_text)
    monkeypatch.setattr("worker.system_inventory._detect_cgroup_version", lambda: "v2")

    inventory = collect_static_inventory()
    assert inventory.hardware["gpu"]["container_visible"]["count"] is None
    assert any("gpu_inventory_unavailable" in warning for warning in inventory.warnings)


def test_collect_static_inventory_reads_cgroups_v2(monkeypatch) -> None:
    def _run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "nvidia-smi":
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="NVIDIA RTX 5080, 16384, 550.54.14, 12.4\n",
                stderr="",
            )
        if cmd[0] == "ffmpeg":
            return subprocess.CompletedProcess(cmd, 0, stdout="ffmpeg version 6.1\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def _safe_read_text(path):
        normalized = str(path).replace("\\", "/")
        mapping = {
            "/proc/cpuinfo": "model name\t: Test CPU v2\n",
            "/proc/meminfo": "MemTotal: 2000 kB\nMemAvailable: 1000 kB\n",
            "/proc/mounts": "overlay / overlay rw 0 0\n",
            "/etc/os-release": 'PRETTY_NAME="TestOS v2"\n',
            "/proc/1/cgroup": "0::/\n",
            "/sys/fs/cgroup/cpu.max": "250000 100000\n",
            "/sys/fs/cgroup/memory.max": "2147483648\n",
        }
        return mapping.get(normalized)

    monkeypatch.setattr("worker.system_inventory._run_command", _run_command)
    monkeypatch.setattr("worker.system_inventory._safe_read_text", _safe_read_text)
    monkeypatch.setattr("worker.system_inventory._detect_cgroup_version", lambda: "v2")

    inventory = collect_static_inventory()
    assert inventory.hardware["cpu"]["container_visible"]["cgroup_quota_cores"] == 2.5
    assert inventory.hardware["memory"]["container_visible"]["cgroup_limit_bytes"] == 2147483648
    assert inventory.hardware["gpu"]["container_visible"]["count"] == 1


def test_collect_static_inventory_reads_cgroups_v1(monkeypatch) -> None:
    def _run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "nvidia-smi":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "ffmpeg":
            return subprocess.CompletedProcess(cmd, 0, stdout="ffmpeg version 6.1\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def _safe_read_text(path):
        normalized = str(path).replace("\\", "/")
        mapping = {
            "/proc/cpuinfo": "model name\t: Test CPU v1\n",
            "/proc/meminfo": "MemTotal: 2000 kB\nMemAvailable: 1000 kB\n",
            "/proc/mounts": "overlay / overlay rw 0 0\n",
            "/etc/os-release": 'PRETTY_NAME="TestOS v1"\n',
            "/proc/1/cgroup": "10:cpu:/docker\n",
            "/sys/fs/cgroup/cpu.max": None,
            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "150000\n",
            "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000\n",
            "/sys/fs/cgroup/memory.max": None,
            "/sys/fs/cgroup/memory/memory.limit_in_bytes": "1073741824\n",
        }
        return mapping.get(normalized)

    monkeypatch.setattr("worker.system_inventory._run_command", _run_command)
    monkeypatch.setattr("worker.system_inventory._safe_read_text", _safe_read_text)
    monkeypatch.setattr("worker.system_inventory._detect_cgroup_version", lambda: "v1")

    inventory = collect_static_inventory()
    assert inventory.hardware["cpu"]["container_visible"]["cgroup_quota_cores"] == 1.5
    assert inventory.hardware["memory"]["container_visible"]["cgroup_limit_bytes"] == 1073741824


def test_collect_static_inventory_partial_metrics_do_not_fail(monkeypatch) -> None:
    def _run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "nvidia-smi":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")
        if cmd[0] == "ffmpeg":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no ffmpeg")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def _safe_read_text(path):
        normalized = str(path).replace("\\", "/")
        mapping = {
            "/proc/cpuinfo": None,
            "/proc/meminfo": None,
            "/proc/mounts": None,
            "/etc/os-release": None,
            "/proc/1/cgroup": "",
            "/sys/fs/cgroup/cpu.max": "oops\n",
            "/sys/fs/cgroup/memory.max": "not-a-number\n",
        }
        return mapping.get(normalized)

    monkeypatch.setattr("worker.system_inventory._run_command", _run_command)
    monkeypatch.setattr("worker.system_inventory._safe_read_text", _safe_read_text)
    monkeypatch.setattr("worker.system_inventory._detect_cgroup_version", lambda: None)

    inventory = collect_static_inventory()
    assert isinstance(inventory.hardware, dict)
    assert isinstance(inventory.software, dict)
    assert any("gpu_inventory_unavailable" in warning for warning in inventory.warnings)
    assert any("memory_inventory_partial" in warning for warning in inventory.warnings)
    assert any("software_inventory_partial" in warning for warning in inventory.warnings)
