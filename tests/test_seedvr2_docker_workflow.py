from pathlib import Path

WORKFLOW_PATH = Path(".github/workflows/seedvr2-docker.yml")


def test_seedvr2_docker_workflow_supports_seedvr2_branch_push_and_dispatch() -> None:
    content = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "push:" in content
    assert '- "seedvr2/**"' in content
    assert "workflow_dispatch:" in content
    assert '- ".github/workflows/seedvr2-docker.yml"' in content


def test_seedvr2_docker_workflow_uses_dispatch_inputs_only_when_available() -> None:
    content = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "github.event_name == 'workflow_dispatch' && inputs.dockerfile || 'prototype/seedvr2/Dockerfile'" in content
    assert "github.event_name == 'workflow_dispatch' && inputs.image_name || 'vast-seedvr2-proto:ci-smoke'" in content


def test_seedvr2_docker_workflow_runs_cli_and_venv_smokes() -> None:
    content = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "Smoke run SeedVR2 prototype CLI help" in content
    assert 'docker run --rm "$IMAGE_NAME" --help' in content
    assert "Verify venv python and torch/cuda runtime" in content
    assert 'docker run --rm --entrypoint python "$IMAGE_NAME" -c "import os, sys, torch, torchvision;' in content
