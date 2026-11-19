from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vast_upscaler.preprocessing_planner import CropRect, PreprocessingPlan


class RemotePreprocessingContractError(RuntimeError):
    pass


class RemotePreprocessingContractValidationError(RemotePreprocessingContractError):
    pass


class RemoteCropParams(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class RemoteDeinterlaceParams(BaseModel):
    algorithm: Literal["bwdif"] = "bwdif"
    mode: Literal["send_frame"] = "send_frame"
    parity: Literal["auto"] = "auto"
    deint: Literal["all"] = "all"


class RemoteOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Literal["crop", "deinterlace"]
    params: RemoteCropParams | RemoteDeinterlaceParams


class RemotePreprocessingContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["1.0"]
    operations: list[RemoteOperation] = Field(default_factory=list)


def load_preprocessing_plan_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RemotePreprocessingContractError(f"Unable to read preprocessing plan: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RemotePreprocessingContractError(f"Invalid preprocessing plan JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RemotePreprocessingContractValidationError("Preprocessing plan root must be a JSON object.")
    return payload


def build_remote_preprocessing_contract_from_plan_payload(plan_payload: dict[str, Any]) -> RemotePreprocessingContract:
    try:
        plan = PreprocessingPlan.model_validate(plan_payload)
    except ValidationError as exc:
        raise RemotePreprocessingContractValidationError(f"Invalid preprocessing plan structure: {exc}") from exc
    return build_remote_preprocessing_contract_from_plan(plan)


def build_remote_preprocessing_contract_from_plan(plan: PreprocessingPlan) -> RemotePreprocessingContract:
    if plan.version != "1.0":
        raise RemotePreprocessingContractValidationError(
            f"Unsupported preprocessing plan version '{plan.version}'. Expected '1.0'."
        )

    source_width = plan.source_summary.width
    source_height = plan.source_summary.height
    if source_width is None or source_height is None:
        raise RemotePreprocessingContractValidationError(
            "Preprocessing plan must provide source_summary.width and source_summary.height."
        )

    operations: list[RemoteOperation] = []
    seen_enabled_ids: set[str] = set()

    for operation in plan.operations:
        operation_id = operation.operation_id
        if operation_id not in {"crop", "deinterlace", "denoise", "deblock"}:
            raise RemotePreprocessingContractValidationError(f"Unknown preprocessing operation_id '{operation_id}'.")
        if not operation.enabled:
            continue
        if operation_id in seen_enabled_ids:
            raise RemotePreprocessingContractValidationError(f"Duplicate enabled operation '{operation_id}'.")
        seen_enabled_ids.add(operation_id)

        if operation_id == "crop":
            operations.append(_convert_crop_operation(operation.parameters, source_width=source_width, source_height=source_height))
            continue
        if operation_id == "deinterlace":
            operations.append(_convert_deinterlace_operation(operation.parameters))
            continue
        raise RemotePreprocessingContractValidationError(
            f"Operation '{operation_id}' is enabled but not supported for remote execution in this phase."
        )

    return RemotePreprocessingContract(version="1.0", operations=operations)


def dump_remote_preprocessing_contract_payload(contract: RemotePreprocessingContract) -> dict[str, Any]:
    return contract.model_dump(mode="json")


def _convert_crop_operation(parameters: dict[str, Any], *, source_width: int, source_height: int) -> RemoteOperation:
    mode = parameters.get("mode")
    if mode not in {"auto_crop", "custom_crop"}:
        raise RemotePreprocessingContractValidationError("Enabled crop operation requires mode auto_crop or custom_crop.")

    if parameters.get("preserve_aspect_ratio") is not True:
        raise RemotePreprocessingContractValidationError("Enabled crop operation requires preserve_aspect_ratio=true.")
    if parameters.get("allow_stretch") is not False:
        raise RemotePreprocessingContractValidationError("Enabled crop operation requires allow_stretch=false.")

    crop_rect_payload = parameters.get("crop_rect")
    if not isinstance(crop_rect_payload, dict):
        raise RemotePreprocessingContractValidationError("Enabled crop operation requires a crop_rect object.")
    try:
        rect = CropRect.model_validate(crop_rect_payload)
    except ValidationError as exc:
        raise RemotePreprocessingContractValidationError(f"Invalid crop_rect payload: {exc}") from exc

    if rect.x + rect.width > source_width or rect.y + rect.height > source_height:
        raise RemotePreprocessingContractValidationError("Crop rectangle is out of source bounds.")
    if rect.width % 2 != 0 or rect.height % 2 != 0 or rect.x % 2 != 0 or rect.y % 2 != 0:
        raise RemotePreprocessingContractValidationError(
            "Crop rectangle must use even x/y/width/height for yuv420-compatible processing."
        )

    return RemoteOperation(id="crop", params=RemoteCropParams(**rect.model_dump(mode="json")))


def _convert_deinterlace_operation(parameters: dict[str, Any]) -> RemoteOperation:
    mode = parameters.get("mode")
    if mode != "enable":
        raise RemotePreprocessingContractValidationError("Enabled deinterlace operation requires mode=enable.")
    return RemoteOperation(id="deinterlace", params=RemoteDeinterlaceParams())
