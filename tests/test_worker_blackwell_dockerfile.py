from pathlib import Path

DOCKERFILE_PATH = Path("worker/Dockerfile.blackwell")
REQUIREMENTS_PATH = Path("worker/requirements.blackwell.txt")


def test_blackwell_dockerfile_uses_newer_cuda_stack() -> None:
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime" in content
    assert "EXPECTED_TORCH_VERSION=2.14.1" in content
    assert "EXPECTED_TORCHVISION_VERSION=0.29.1" in content
    assert "EXPECTED_CUDA_VERSION=13.0" in content
    assert "requirements.blackwell.txt" in content


def test_blackwell_dockerfile_uses_venv_for_pep668_compatibility() -> None:
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "python3.12-venv" in content
    assert "command -v python" in content
    assert "command -v python3" in content
    assert "[python-diagnostics] python sys.executable=" in content
    assert "[python-diagnostics] python3 sys.executable=" in content
    assert "selected_base_python=$BASE_PY" in content
    assert "No base Python interpreter with preinstalled torch found in this image" in content
    assert "-m venv --system-site-packages ${WORKER_VENV}" in content
    assert "ENV PATH=${WORKER_VENV}/bin:${PATH}" in content
    assert "python -m pip install --upgrade pip" in content
    assert "--break-system-packages" not in content
    assert "\nRUN pip install --upgrade pip" not in content


def test_blackwell_dockerfile_keeps_basicsr_patch_logic() -> None:
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "functional_tensor" in content
    assert "torchvision.transforms.functional import rgb_to_grayscale" in content
    assert "expected BasicSR patch target not found" in content


def test_blackwell_requirements_pin_torch_torchvision_pair() -> None:
    content = REQUIREMENTS_PATH.read_text(encoding="utf-8")
    assert "torch==2.14.1" in content
    assert "torchvision==0.29.1" in content
