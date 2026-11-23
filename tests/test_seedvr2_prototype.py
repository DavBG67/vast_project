from __future__ import annotations

import pytest

from vast_upscaler.seedvr2_prototype import (
    PROFILE_REFERENCE,
    PROFILE_SMOKE,
    PROFILES,
    detect_oom_failure,
    parse_timecode_to_seconds,
)


def test_seedvr2_profiles_cover_smoke_and_reference() -> None:
    assert set(PROFILES.keys()) == {"smoke", "reference"}
    assert PROFILE_SMOKE.excerpt_duration_seconds <= 3.0
    assert PROFILE_REFERENCE.target_width == 1280
    assert PROFILE_REFERENCE.target_height == 720


def test_parse_timecode_to_seconds_accepts_mmss_and_hhmmss() -> None:
    assert parse_timecode_to_seconds("00:39") == 39.0
    assert parse_timecode_to_seconds("01:02:03") == 3723.0


def test_parse_timecode_to_seconds_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        parse_timecode_to_seconds("")
    with pytest.raises(ValueError):
        parse_timecode_to_seconds("99")
    with pytest.raises(ValueError):
        parse_timecode_to_seconds("00:60")


def test_detect_oom_failure_matches_common_cuda_errors() -> None:
    assert detect_oom_failure("RuntimeError: CUDA out of memory while allocating tensor")
    assert detect_oom_failure("CUBLAS_STATUS_ALLOC_FAILED")
    assert not detect_oom_failure("Completed successfully")

