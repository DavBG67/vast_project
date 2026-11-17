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
