from pathlib import Path

import pytest

from vast_upscaler.preprocessing_planner import (
    PlannerUserChoices,
    PreprocessingPlannerValidationError,
    build_preprocessing_plan,
)


def _analysis_payload(
    *,
    width: int = 1280,
    height: int = 720,
    idet_classification: str = "progressive",
    idet_confidence: str = "high",
    crop_confidence: str = "high",
    crop_rect: dict | None = None,
    crop_margins: dict | None = None,
    partial: bool = False,
    likely_noise: str = "unknown",
    likely_compression: str = "unknown",
) -> dict:
    return {
        "file_path": str(Path("sample.mp4")),
        "width": width,
        "height": height,
        "frame_rate_fps": 25.0,
        "duration_seconds": 12.0,
        "video_codec": "h264",
        "interlace": {"field_order": "progressive", "interlaced": False, "confidence": "certain", "note": "meta"},
        "analysis_report": {
            "version": "1.0",
            "visual_diagnostics": {
                "partial": partial,
                "cropdetect": {
                    "status": "ok",
                    "confidence": crop_confidence,
                    "recommended_crop": crop_rect or {"x": 0, "y": 20, "width": width, "height": height - 40},
                    "probable_margins": crop_margins or {"top": 20, "bottom": 20, "left": 0, "right": 0},
                },
                "idet": {
                    "status": "ok",
                    "confidence": idet_confidence,
                    "classification": idet_classification,
                },
                "noise_artifacts": {
                    "status": "ok",
                    "confidence": "low",
                    "likely_noise": likely_noise,
                    "likely_compression_artifacts": likely_compression,
                },
            },
        },
    }


def test_progressive_no_black_bars_keeps_crop_disabled_by_default() -> None:
    payload = _analysis_payload(
        crop_margins={"top": 0, "bottom": 0, "left": 0, "right": 0},
        crop_rect={"x": 0, "y": 0, "width": 1280, "height": 720},
        crop_confidence="medium",
    )
    plan = build_preprocessing_plan(analysis_payload=payload)
    crop = next(op for op in plan.operations if op.operation_id == "crop")
    assert crop.enabled is False
    assert crop.parameters["mode"] == "preserve"


def test_black_bars_high_confidence_recommended_but_not_enabled() -> None:
    payload = _analysis_payload()
    plan = build_preprocessing_plan(analysis_payload=payload)
    crop = next(op for op in plan.operations if op.operation_id == "crop")
    assert crop.recommended is True
    assert crop.enabled is False


def test_incoherent_crop_diagnostics_disable_auto_crop() -> None:
    payload = _analysis_payload(crop_confidence="low")
    plan = build_preprocessing_plan(
        analysis_payload=payload,
        user_choices=PlannerUserChoices(crop_mode="auto_crop"),
    )
    crop = next(op for op in plan.operations if op.operation_id == "crop")
    assert crop.enabled is False
    assert any("auto_crop selected with low/unknown crop confidence" in warning for warning in plan.warnings)


def test_interlace_confirmed_recommends_deinterlace_but_not_auto_enabled() -> None:
    payload = _analysis_payload(idet_classification="interlaced", idet_confidence="high")
    plan = build_preprocessing_plan(analysis_payload=payload)
    deint = next(op for op in plan.operations if op.operation_id == "deinterlace")
    assert deint.recommended is True
    assert deint.enabled is False


def test_interlace_ambiguous_does_not_recommend_enable() -> None:
    payload = _analysis_payload(idet_classification="mixed", idet_confidence="low")
    plan = build_preprocessing_plan(analysis_payload=payload)
    deint = next(op for op in plan.operations if op.operation_id == "deinterlace")
    assert deint.recommended is False


def test_noise_unknown_keeps_denoise_unknown_and_disabled() -> None:
    payload = _analysis_payload(likely_noise="unknown")
    plan = build_preprocessing_plan(analysis_payload=payload)
    denoise = next(op for op in plan.operations if op.operation_id == "denoise")
    assert denoise.enabled is False
    assert denoise.recommended is False


def test_partial_report_surfaces_plan_warning() -> None:
    payload = _analysis_payload(partial=True)
    plan = build_preprocessing_plan(analysis_payload=payload)
    assert any("partial" in warning.lower() for warning in plan.warnings)


def test_custom_crop_valid_is_enabled_with_validated_rect() -> None:
    payload = _analysis_payload(width=1920, height=1080)
    plan = build_preprocessing_plan(
        analysis_payload=payload,
        user_choices=PlannerUserChoices(
            crop_mode="custom_crop",
            custom_crop_margins={"top": 10, "bottom": 10, "left": 20, "right": 20},
        ),
    )
    crop = next(op for op in plan.operations if op.operation_id == "crop")
    assert crop.enabled is True
    assert crop.parameters["crop_rect"]["width"] == 1880
    assert crop.parameters["crop_rect"]["height"] == 1060
    assert crop.parameters["allow_stretch"] is False


def test_custom_crop_invalid_is_rejected() -> None:
    payload = _analysis_payload(width=640, height=360)
    with pytest.raises(PreprocessingPlannerValidationError):
        build_preprocessing_plan(
            analysis_payload=payload,
            user_choices=PlannerUserChoices(
                crop_mode="custom_crop",
                custom_crop_margins={"top": 0, "bottom": 0, "left": 400, "right": 400},
            ),
        )


def test_analysis_without_visual_diagnostics_is_supported() -> None:
    payload = {
        "file_path": "sample.mp4",
        "width": 1280,
        "height": 720,
        "frame_rate_fps": 30.0,
        "duration_seconds": 10.0,
        "video_codec": "h264",
        "interlace": {"field_order": None, "interlaced": None, "confidence": "uncertain", "note": "n/a"},
    }
    plan = build_preprocessing_plan(analysis_payload=payload)
    assert plan.source_summary.diagnostics_available is False
    assert any("unavailable" in warning.lower() for warning in plan.warnings)
