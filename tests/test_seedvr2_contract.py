from __future__ import annotations

from pathlib import Path

from vast_upscaler.remote_seedvr2_worker import _build_seedvr2_command
from vast_upscaler.seedvr2_prototype import PROFILES, build_parser

DOCKERFILE_PATH = Path("prototype/seedvr2/Dockerfile")


def test_remote_seedvr2_cli_arguments_exist_in_runner_parser() -> None:
    command = _build_seedvr2_command(
        input_path="/workspace/seedvr2-sessions/job/input/input.mp4",
        profile="smoke",
        output_root="/workspace/seedvr2-sessions/job/output",
        work_root="/workspace/seedvr2-sessions/job/work",
        cache_dir="/workspace/seedvr2-sessions/job/cache",
    )
    parser = build_parser()
    parser_options = set(parser._option_string_actions.keys())
    command_options = {token for token in command if token.startswith("--")}
    assert command_options <= parser_options


def test_seedvr2_runner_path_and_python_match_docker_contract() -> None:
    dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert 'ARG VENV_PATH=/opt/seedvr2-venv' in dockerfile
    assert 'ENTRYPOINT ["python", "/opt/prototype/seedvr2_prototype.py"]' in dockerfile
    command = _build_seedvr2_command(
        input_path="/workspace/seedvr2-sessions/job/input/input.mp4",
        profile="reference",
        output_root="/workspace/seedvr2-sessions/job/output",
        work_root="/workspace/seedvr2-sessions/job/work",
        cache_dir="/workspace/seedvr2-sessions/job/cache",
    )
    assert command[0] == "/opt/seedvr2-venv/bin/python"
    assert command[1] == "/opt/prototype/seedvr2_prototype.py"
    assert "--input-is-excerpt" in command


def test_seedvr2_profiles_used_by_web_are_available_in_runner() -> None:
    assert {"smoke", "reference"} <= set(PROFILES.keys())
