from __future__ import annotations

from pathlib import Path

DOCKERFILE_PATH = Path("prototype/seedvr2/Dockerfile")
CONSTRAINTS_PATH = Path("prototype/seedvr2/requirements.constraints.txt")


def test_seedvr2_prototype_dockerfile_isolated_from_stable_worker() -> None:
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "COPY worker/" not in content
    assert "worker/process.py" not in content
    assert "python3-venv" in content
    assert "COPY src/vast_upscaler/seedvr2_prototype.py" in content
    assert 'ENTRYPOINT ["python", "/opt/prototype/seedvr2_prototype.py"]' in content
    assert "ARG VENV_PATH=/opt/seedvr2-venv" in content
    assert "PATH=${VENV_PATH}/bin:${PATH}" in content
    assert "python -m venv --system-site-packages" in content
    assert "sys.executable.startswith(os.environ['VENV_PATH'])" in content


def test_seedvr2_prototype_dockerfile_pins_runtime_and_seedvr2_source_ref() -> None:
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime" in content
    assert "ARG NUMZ_REPO_REF=4490bd1f482e026674543386bb2a4d176da245b9" in content
    assert "EXPECTED_TORCH_VERSION=2.14.1" in content
    assert "EXPECTED_TORCHVISION_VERSION=0.29.1" in content
    assert "EXPECTED_CUDA_VERSION=13.0" in content
    assert "grep -Ev '^(torch|torchvision)" in content
    assert '-r /tmp/requirements.seedvr2.txt' in content
    assert '"${VENV_PATH}/bin/python" /opt/seedvr2/inference_cli.py --help > /tmp/seedvr2-help.txt' in content
    assert "grep -q -- '--dit_model' /tmp/seedvr2-help.txt" in content


def test_seedvr2_prototype_constraints_lock_direct_dependencies() -> None:
    content = CONSTRAINTS_PATH.read_text(encoding="utf-8")
    assert "torch==2.14.1" in content
    assert "torchvision==0.29.1" in content
    assert "numpy==2.2.6" in content
    assert "diffusers==0.33.1" in content
    assert "gguf==0.17.1" in content

