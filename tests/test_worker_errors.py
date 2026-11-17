from pathlib import Path

import pytest

from worker.process import WorkerError, _ensure_disk_space


def test_ensure_disk_space_raises(monkeypatch, tmp_path: Path) -> None:
    class Usage:
        total = 100
        used = 99
        free = 1

    monkeypatch.setattr("worker.process.shutil.disk_usage", lambda _p: Usage())

    with pytest.raises(WorkerError):
        _ensure_disk_space(tmp_path, min_free_gb=0.1)
