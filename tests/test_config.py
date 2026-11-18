from pathlib import Path

from vast_upscaler.config import Settings, load_config


def test_load_config_defaults_when_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    settings = load_config(config_path)
    assert isinstance(settings, Settings)
    assert settings.video.target_width == 1920
    assert settings.vast.max_hourly_price_usd == 0.30


def test_load_config_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
video:
  target_width: 1280
  target_height: 720
  default_profile: fast
vast:
  max_hourly_price_usd: 0.22
  gpu_models:
    - RTX_4090
""".strip(),
        encoding="utf-8",
    )

    settings = load_config(config_path)
    assert settings.video.target_width == 1280
    assert settings.video.target_height == 720
    assert settings.video.default_profile == "fast"
    assert settings.vast.max_hourly_price_usd == 0.22
    assert settings.vast.gpu_models == ["RTX_4090"]


def test_load_config_reads_local_dotenv_when_env_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.delenv("VAST_SSH_PRIVATE_KEY", raising=False)

    config_path = tmp_path / "config.yaml"
    (tmp_path / ".env").write_text(
        "VAST_API_KEY=dotenv-api-key\nVAST_SSH_PRIVATE_KEY=C:/keys/id_ed25519\n",
        encoding="utf-8",
    )

    settings = load_config(config_path)

    assert settings.secrets.vast_api_key == "dotenv-api-key"
    assert settings.secrets.ssh_private_key_path == "C:/keys/id_ed25519"


def test_load_config_keeps_system_env_priority_over_dotenv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VAST_API_KEY", "system-api-key")
    monkeypatch.delenv("VAST_SSH_PRIVATE_KEY", raising=False)

    config_path = tmp_path / "config.yaml"
    (tmp_path / ".env").write_text(
        "VAST_API_KEY=dotenv-api-key\nVAST_SSH_PRIVATE_KEY=C:/keys/from-dotenv\n",
        encoding="utf-8",
    )

    settings = load_config(config_path)

    assert settings.secrets.vast_api_key == "system-api-key"
    assert settings.secrets.ssh_private_key_path == "C:/keys/from-dotenv"
