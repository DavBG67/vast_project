from pathlib import Path

WORKFLOW_PATH = Path(".github/workflows/docker-worker.yml")


def test_blackwell_dispatch_uses_blackwell_expected_versions_when_dispatch_defaults_are_left() -> None:
    content = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert 'default: "2.14.1"' in content
    assert 'default: "0.29.1"' in content
    assert 'default: "13.0"' in content
    assert 'default: "sm_120"' in content


def test_compatibility_check_logs_expected_and_detected_versions() -> None:
    content = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "[compat] torch expected=" in content
    assert "[compat] torchvision expected=" in content
    assert "[compat] cuda expected=" in content
    assert "unexpected torch public version: expected" in content
    assert "unexpected torchvision public version: expected" in content
    assert "unexpected compiled cuda version: expected" in content
    assert "[compat] expected_arch_token=" in content
    assert "expected arch token" in content
