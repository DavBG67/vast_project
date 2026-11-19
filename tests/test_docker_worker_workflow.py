from pathlib import Path

WORKFLOW_PATH = Path(".github/workflows/docker-worker.yml")


def test_blackwell_dispatch_uses_blackwell_expected_versions_when_dispatch_defaults_are_left() -> None:
    content = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "inputs.dockerfile == 'worker/Dockerfile.blackwell'" in content
    assert "inputs.expected_torch_version == '2.4.1' && '2.14.1'" in content
    assert "inputs.expected_torchvision_version == '0.19.1' && '0.29.1'" in content
    assert "inputs.expected_cuda_version == '12.1' && '13.0'" in content


def test_compatibility_check_logs_expected_and_detected_versions() -> None:
    content = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "[compat] torch expected=" in content
    assert "[compat] torchvision expected=" in content
    assert "[compat] cuda expected=" in content
    assert "unexpected torch public version: expected" in content
    assert "unexpected torchvision public version: expected" in content
    assert "unexpected compiled cuda version: expected" in content
