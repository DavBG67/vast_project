from pathlib import Path

DOCKERFILE_PATH = Path("worker/Dockerfile")
PATCH_COMMENT = "# Patch BasicSR import for torchvision>=0.19 where functional_tensor module path is removed."


def test_basicsr_patch_uses_find_spec_without_importing_basicsr() -> None:
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert PATCH_COMMENT in content

    line = next(
        ln for ln in content.splitlines() if "find_spec('basicsr')" in ln and "degradations.py" in ln
    )

    assert "import importlib.util" in line
    assert "find_spec('basicsr')" in line
    assert "submodule_search_locations" in line
    assert "import basicsr" not in line
    assert "expected BasicSR patch target not found" in line


def test_worker_dockerfile_pins_blackwell_compatible_runtime_stack() -> None:
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime" in content
    assert "ARG EXPECTED_TORCH_VERSION=2.14.1" in content
    assert "ARG EXPECTED_TORCHVISION_VERSION=0.29.1" in content
    assert "ARG EXPECTED_CUDA_VERSION=13.0" in content
    assert "torchvision==${EXPECTED_TORCHVISION_VERSION}" in content
