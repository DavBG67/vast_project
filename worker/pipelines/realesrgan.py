from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class RealesrganError(RuntimeError):
    pass


@dataclass(frozen=True)
class RealesrganConfig:
    weights_path: Path
    model_name: str
    model_scale: int
    tile: int
    tile_pad: int
    pre_pad: int


class RealesrganUpscaler:
    def __init__(self, config: RealesrganConfig, device: str = "cuda") -> None:
        self._config = config
        self._device = device
        self._upsampler = None

    def load(self) -> None:
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except Exception as exc:
            raise RealesrganError(f"Unable to import Real-ESRGAN runtime: {exc}") from exc

        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise RealesrganError("CUDA is not available. Real-ESRGAN GPU pipeline requires NVIDIA CUDA.")

        if not self._config.weights_path.exists():
            raise RealesrganError(f"Model weights file not found: {self._config.weights_path}")

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=self._config.model_scale,
        )

        self._upsampler = RealESRGANer(
            scale=self._config.model_scale,
            model_path=str(self._config.weights_path),
            model=model,
            tile=self._config.tile,
            tile_pad=self._config.tile_pad,
            pre_pad=self._config.pre_pad,
            half=True,
            gpu_id=0,
        )

    def upscale_frame(self, frame_bgr):
        if self._upsampler is None:
            raise RealesrganError("Upscaler not loaded")
        try:
            output, _ = self._upsampler.enhance(frame_bgr, outscale=float(self._config.model_scale))
            return output
        except RuntimeError as exc:
            msg = str(exc)
            if "out of memory" in msg.lower():
                raise RealesrganError("CUDA out of memory during Real-ESRGAN inference") from exc
            raise RealesrganError(f"Real-ESRGAN inference failed: {exc}") from exc

    @staticmethod
    def detect_gpu_name() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except (ImportError, RuntimeError, AttributeError, OSError):
            return "unknown"
        return "unknown"
