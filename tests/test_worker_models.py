from pathlib import Path

import pytest

from worker.models import ProfileNotSupportedError, select_model


def test_select_model_balanced() -> None:
    model = select_model("balanced", Path("/opt/models"))
    assert model.name == "realesrgan"
    assert model.model_name == "RealESRGAN_x4plus"
    assert model.model_scale == 4


def test_select_model_unsupported() -> None:
    with pytest.raises(ProfileNotSupportedError):
        select_model("fast", Path("/opt/models"))
