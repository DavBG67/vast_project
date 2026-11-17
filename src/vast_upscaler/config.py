from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError


class VastConfig(BaseModel):
    gpu_models: list[str] = Field(default_factory=lambda: ["RTX 3090 Ti", "RTX 3090", "RTX 4090"])
    max_hourly_price_usd: float = 0.30
    min_reliability: float = 0.99
    verified_only: bool = True
    disk_gb: int = 50
    max_session_hours: int = 12


class VideoConfig(BaseModel):
    target_width: int = 1920
    target_height: int = 1080
    default_profile: Literal["fast", "balanced", "quality"] = "balanced"
    output_container: Literal["mkv", "mp4"] = "mkv"
    video_codec: Literal["hevc", "h264", "av1"] = "hevc"


class TransferConfig(BaseModel):
    method: Literal["sftp"] = "sftp"
    verify_checksum: bool = True


class SecretsConfig(BaseModel):
    vast_api_key: str | None = Field(default_factory=lambda: os.getenv("VAST_API_KEY"))
    ssh_private_key_path: str | None = Field(default_factory=lambda: os.getenv("VAST_SSH_PRIVATE_KEY"))


class Settings(BaseModel):
    vast: VastConfig = Field(default_factory=VastConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    transfer: TransferConfig = Field(default_factory=TransferConfig)
    secrets: SecretsConfig = Field(default_factory=SecretsConfig)


class ConfigError(RuntimeError):
    pass


def load_config(config_path: Path | None = None) -> Settings:
    path = config_path or Path("config.yaml")
    if not path.exists():
        return Settings()

    try:
        raw_data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ConfigError(f"Unable to read config file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in config file: {path}") from exc

    if not isinstance(raw_data, dict):
        raise ConfigError("Config root must be a mapping/object")

    try:
        return Settings(**raw_data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config values in {path}: {exc}") from exc
