from __future__ import annotations

import copy

import pytest

from vast_upscaler.preprocessing_contract import (
    RemotePreprocessingContractValidationError,
    build_remote_preprocessing_contract_from_plan_payload,
)
from vast_upscaler.preprocessing_planner import (
    PlannerUserChoices,
    build_preprocessing_plan,
)


def _analysis_payload() -> dict:
    return {
        "file_path": "sample.mp4",
        "width": 640,
        "height": 480,
        "frame_rate_fps": 25.0,
        "duration_seconds": 10.0,
        "video_codec": "h264",
        "interlace": {"field_order": "progressive", "interlaced": False, "confidence": "certain", "note": "meta"},
    }


def test_build_remote_contract_filters_disabled_operations() -> None:
    plan = build_preprocessing_plan(analysis_payload=_analysis_payload())
    payload = plan.model_dump(mode="json")

    contract = build_remote_preprocessing_contract_from_plan_payload(payload)

    assert contract.version == "1.0"
    assert contract.operations == []


def test_build_remote_contract_converts_enabled_crop_without_mutating_source_payload() -> None:
    plan = build_preprocessing_plan(
        analysis_payload=_analysis_payload(),
        user_choices=PlannerUserChoices(crop_mode="custom_crop", custom_crop_margins={"top": 0, "bottom": 0, "left": 8, "right": 8}),
    )
    payload = plan.model_dump(mode="json")
    original = copy.deepcopy(payload)

    contract = build_remote_preprocessing_contract_from_plan_payload(payload)

    assert len(contract.operations) == 1
    assert contract.operations[0].id == "crop"
    assert contract.operations[0].params.width == 624
    assert payload == original


def test_build_remote_contract_rejects_unknown_operation() -> None:
    plan = build_preprocessing_plan(analysis_payload=_analysis_payload())
    payload = plan.model_dump(mode="json")
    payload["operations"].append(
        {
            "operation_id": "mystery",
            "kind": "crop",
            "enabled": False,
            "recommended": False,
            "confidence": "unknown",
            "parameters": {},
            "justification": "n/a",
            "risk_notes": [],
        }
    )

    with pytest.raises(RemotePreprocessingContractValidationError):
        build_remote_preprocessing_contract_from_plan_payload(payload)


def test_build_remote_contract_rejects_enabled_unsupported_operation() -> None:
    plan = build_preprocessing_plan(
        analysis_payload=_analysis_payload(),
        user_choices=PlannerUserChoices(denoise_mode="enable"),
    )
    payload = plan.model_dump(mode="json")

    with pytest.raises(RemotePreprocessingContractValidationError):
        build_remote_preprocessing_contract_from_plan_payload(payload)


def test_build_remote_contract_rejects_incompatible_version() -> None:
    plan = build_preprocessing_plan(analysis_payload=_analysis_payload())
    payload = plan.model_dump(mode="json")
    payload["version"] = "2.0"

    with pytest.raises(RemotePreprocessingContractValidationError):
        build_remote_preprocessing_contract_from_plan_payload(payload)


def test_build_remote_contract_rejects_invalid_crop_params() -> None:
    plan = build_preprocessing_plan(
        analysis_payload=_analysis_payload(),
        user_choices=PlannerUserChoices(crop_mode="custom_crop", custom_crop_margins={"top": 0, "bottom": 0, "left": 7, "right": 7}),
    )
    payload = plan.model_dump(mode="json")

    with pytest.raises(RemotePreprocessingContractValidationError):
        build_remote_preprocessing_contract_from_plan_payload(payload)
