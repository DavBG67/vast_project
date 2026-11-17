from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class ProfileNotSupportedError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_name: str
    model_scale: int
    weights_path: Path
    tile: int
    tile_pad: int
    pre_pad: int


def select_model(profile: str, models_dir: Path) -> ModelSpec:
    normalized = profile.strip().lower()
    if normalized != "balanced":
        raise ProfileNotSupportedError(
            f"Profile '{profile}' is not available in phase 3. Supported profile: balanced."
        )

    return ModelSpec(
        name="realesrgan",
        model_name="RealESRGAN_x4plus",
        model_scale=4,
        weights_path=models_dir / "RealESRGAN_x4plus.pth",
        tile=256,
        tile_pad=10,
        pre_pad=0,
    )
