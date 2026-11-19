from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


class PreprocessingPlannerError(RuntimeError):
    pass


class PreprocessingPlannerValidationError(PreprocessingPlannerError):
    pass


class CropRect(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class CropMargins(BaseModel):
    top: int = Field(ge=0)
    bottom: int = Field(ge=0)
    left: int = Field(ge=0)
    right: int = Field(ge=0)


class PlannerUserChoices(BaseModel):
    crop_mode: Literal["preserve", "auto_crop", "custom_crop"] = "preserve"
    custom_crop_margins: CropMargins | None = None
    deinterlace_mode: Literal["preserve", "enable", "disable"] = "preserve"
    denoise_mode: Literal["preserve", "enable", "disable"] = "preserve"
    deblock_mode: Literal["preserve", "enable", "disable"] = "preserve"


class PreprocessingOperation(BaseModel):
    operation_id: str
    kind: Literal["crop", "deinterlace", "denoise", "deblock"]
    enabled: bool
    recommended: bool
    confidence: Literal["high", "medium", "low", "unknown"]
    parameters: dict[str, Any]
    justification: str
    risk_notes: list[str] = Field(default_factory=list)


class RecommendationItem(BaseModel):
    key: Literal["crop", "deinterlace", "denoise", "deblock"]
    recommended_action: str
    confidence: Literal["high", "medium", "low", "unknown"]
    reason: str
    uncertainties: list[str] = Field(default_factory=list)


class EstimatedImpact(BaseModel):
    cpu_cost: Literal["low", "medium", "high", "unknown"]
    gpu_gain: Literal["low", "medium", "high", "unknown"]
    quality_risk: Literal["low", "medium", "high", "unknown"]
    note: str


class SourceSummary(BaseModel):
    file_path: str
    width: int | None = None
    height: int | None = None
    frame_rate_fps: float | None = None
    duration_seconds: float | None = None
    video_codec: str | None = None
    interlace_metadata: dict[str, Any]
    diagnostics_available: bool


class PreprocessingPlan(BaseModel):
    version: str = "1.0"
    source_summary: SourceSummary
    recommendations: list[RecommendationItem]
    user_choices: PlannerUserChoices
    operations: list[PreprocessingOperation]
    warnings: list[str] = Field(default_factory=list)
    estimated_impact: EstimatedImpact


def load_analysis_payload_from_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PreprocessingPlannerError(f"Unable to read analysis payload: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PreprocessingPlannerError(f"Analysis payload is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise PreprocessingPlannerError("Analysis payload root must be a JSON object.")
    return payload


def build_preprocessing_plan(
    *,
    analysis_payload: dict[str, Any],
    user_choices: PlannerUserChoices | None = None,
) -> PreprocessingPlan:
    choices = user_choices or PlannerUserChoices()
    _validate_analysis_payload(analysis_payload)
    source = _build_source_summary(analysis_payload)
    diagnostics = _extract_visual_diagnostics(analysis_payload)
    warnings: list[str] = []

    recommendations = _build_recommendations(analysis_payload, diagnostics)
    crop_op, crop_warnings = _build_crop_operation(
        analysis_payload=analysis_payload,
        diagnostics=diagnostics,
        choices=choices,
    )
    warnings.extend(crop_warnings)
    deint_op, deint_warnings = _build_deinterlace_operation(diagnostics=diagnostics, choices=choices)
    warnings.extend(deint_warnings)
    denoise_op, denoise_warnings = _build_noise_operation(diagnostics=diagnostics, choices=choices)
    warnings.extend(denoise_warnings)
    deblock_op, deblock_warnings = _build_deblock_operation(diagnostics=diagnostics, choices=choices)
    warnings.extend(deblock_warnings)

    if diagnostics is None:
        warnings.append("Visual diagnostics are unavailable; recommendations are conservative.")
    elif diagnostics.get("partial") is True:
        warnings.append("Visual diagnostics are partial; treat recommendations as advisory only.")

    impact = _estimate_impact([crop_op, deint_op, denoise_op, deblock_op])

    return PreprocessingPlan(
        source_summary=source,
        recommendations=recommendations,
        user_choices=choices,
        operations=[crop_op, deint_op, denoise_op, deblock_op],
        warnings=warnings,
        estimated_impact=impact,
    )


def _validate_analysis_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload.get("file_path"), str) or not payload.get("file_path"):
        raise PreprocessingPlannerValidationError("Analysis payload must include a non-empty file_path.")
    width = payload.get("width")
    height = payload.get("height")
    if width is not None and (not isinstance(width, int) or width <= 0):
        raise PreprocessingPlannerValidationError("Analysis payload width must be a positive integer when provided.")
    if height is not None and (not isinstance(height, int) or height <= 0):
        raise PreprocessingPlannerValidationError("Analysis payload height must be a positive integer when provided.")


def _build_source_summary(payload: dict[str, Any]) -> SourceSummary:
    interlace_payload = payload.get("interlace")
    if not isinstance(interlace_payload, dict):
        interlace_payload = {}
    diagnostics = _extract_visual_diagnostics(payload)
    return SourceSummary(
        file_path=payload["file_path"],
        width=payload.get("width"),
        height=payload.get("height"),
        frame_rate_fps=payload.get("frame_rate_fps"),
        duration_seconds=payload.get("duration_seconds"),
        video_codec=payload.get("video_codec"),
        interlace_metadata=interlace_payload,
        diagnostics_available=diagnostics is not None,
    )


def _extract_visual_diagnostics(payload: dict[str, Any]) -> dict[str, Any] | None:
    report = payload.get("analysis_report")
    if not isinstance(report, dict):
        return None
    visual = report.get("visual_diagnostics")
    if not isinstance(visual, dict):
        return None
    return visual


def _build_recommendations(analysis_payload: dict[str, Any], diagnostics: dict[str, Any] | None) -> list[RecommendationItem]:
    crop_rec = RecommendationItem(
        key="crop",
        recommended_action="preserve",
        confidence="unknown",
        reason="No reliable crop recommendation is available.",
        uncertainties=["Visual diagnostics are missing or inconclusive."],
    )
    if diagnostics is not None and isinstance(diagnostics.get("cropdetect"), dict):
        crop = diagnostics["cropdetect"]
        margins = crop.get("probable_margins")
        confidence = str(crop.get("confidence") or "unknown")
        recommended = (
            isinstance(margins, dict)
            and sum(int(margins.get(key, 0)) for key in ("top", "bottom", "left", "right")) >= 8
            and confidence in {"high", "medium"}
        )
        crop_rec = RecommendationItem(
            key="crop",
            recommended_action="consider_auto_crop" if recommended else "preserve",
            confidence=confidence if confidence in {"high", "medium", "low", "unknown"} else "unknown",
            reason=(
                "Black-border margins were detected with usable confidence."
                if recommended
                else "Crop detection did not produce a robust or significant border signal."
            ),
            uncertainties=[
                "Dark scenes, subtitles, and variable aspect-ratio shots can bias cropdetect results."
            ],
        )

    idet_rec = RecommendationItem(
        key="deinterlace",
        recommended_action="unknown",
        confidence="unknown",
        reason="No idet-based interlace diagnosis is available.",
        uncertainties=["Visual diagnostics are missing or inconclusive."],
    )
    if diagnostics is not None and isinstance(diagnostics.get("idet"), dict):
        idet = diagnostics["idet"]
        cls = str(idet.get("classification") or "indeterminate")
        conf = str(idet.get("confidence") or "unknown")
        if cls == "interlaced" and conf in {"high", "medium"}:
            action = "consider_deinterlace"
            reason = "Interlaced cadence is consistently detected by idet."
        elif cls == "progressive" and conf in {"high", "medium"}:
            action = "avoid_deinterlace"
            reason = "Source appears progressive; deinterlacing may degrade detail."
        elif cls == "mixed":
            action = "manual_review"
            reason = "Mixed cadence detected; selective strategy may be required."
        else:
            action = "unknown"
            reason = "Interlace signal is ambiguous or low confidence."
        idet_rec = RecommendationItem(
            key="deinterlace",
            recommended_action=action,
            confidence=conf if conf in {"high", "medium", "low", "unknown"} else "unknown",
            reason=reason,
            uncertainties=["idet can be unstable on short, noisy, or heavily compressed samples."],
        )

    noise_rec = RecommendationItem(
        key="denoise",
        recommended_action="unknown",
        confidence="unknown",
        reason="Noise level cannot be determined confidently from current indicators.",
        uncertainties=["No reliable visual diagnostics are available."],
    )
    deblock_rec = RecommendationItem(
        key="deblock",
        recommended_action="unknown",
        confidence="unknown",
        reason="Compression artifact level cannot be determined confidently from current indicators.",
        uncertainties=["No reliable visual diagnostics are available."],
    )
    if diagnostics is not None and isinstance(diagnostics.get("noise_artifacts"), dict):
        nd = diagnostics["noise_artifacts"]
        conf = str(nd.get("confidence") or "unknown")
        likely_noise = str(nd.get("likely_noise") or "unknown")
        likely_block = str(nd.get("likely_compression_artifacts") or "unknown")
        noise_rec = RecommendationItem(
            key="denoise",
            recommended_action="consider_light_denoise" if likely_noise in {"high", "medium"} else "unknown",
            confidence=conf if conf in {"high", "medium", "low", "unknown"} else "unknown",
            reason=(
                "Signalstats indicators suggest temporal noise-like behavior."
                if likely_noise in {"high", "medium"}
                else "Signalstats indicators are not conclusive for noise."
            ),
            uncertainties=[
                "signalstats does not directly measure perceptual noise and can be confounded by motion."
            ],
        )
        deblock_rec = RecommendationItem(
            key="deblock",
            recommended_action="consider_light_deblock" if likely_block in {"high", "medium"} else "unknown",
            confidence=conf if conf in {"high", "medium", "low", "unknown"} else "unknown",
            reason=(
                "Bitrate-density indicators suggest possible compression artifacts."
                if likely_block in {"high", "medium"}
                else "Compression-artifact indicators are not conclusive."
            ),
            uncertainties=[
                "Bitrate-density is a weak proxy and should not be interpreted as a certainty."
            ],
        )

    return [crop_rec, idet_rec, noise_rec, deblock_rec]


def _build_crop_operation(
    *,
    analysis_payload: dict[str, Any],
    diagnostics: dict[str, Any] | None,
    choices: PlannerUserChoices,
) -> tuple[PreprocessingOperation, list[str]]:
    warnings: list[str] = []
    width = analysis_payload.get("width")
    height = analysis_payload.get("height")
    diag_rect = None
    diag_conf = "unknown"
    if diagnostics is not None and isinstance(diagnostics.get("cropdetect"), dict):
        crop = diagnostics["cropdetect"]
        diag_conf = str(crop.get("confidence") or "unknown")
        recommended_crop = crop.get("recommended_crop")
        if isinstance(recommended_crop, dict):
            try:
                diag_rect = CropRect(**recommended_crop)
            except ValidationError:
                diag_rect = None

    enabled = False
    recommended = bool(diag_rect is not None and diag_conf in {"high", "medium"})
    params: dict[str, Any] = {
        "mode": choices.crop_mode,
        "crop_rect": None,
        "preserve_aspect_ratio": True,
        "allow_stretch": False,
        "fit_policy_after_crop": "letterbox_or_pad",
    }
    justification = "Preserve mode keeps full frame; crop remains disabled unless explicitly selected."
    risk_notes = ["Cropping can remove intended content; validate visually before applying."]

    if choices.crop_mode == "auto_crop":
        if diag_rect is None:
            warnings.append("auto_crop selected but no valid cropdetect rectangle is available; crop remains disabled.")
        elif diag_conf not in {"high", "medium"}:
            warnings.append("auto_crop selected with low/unknown crop confidence; crop remains disabled.")
        else:
            _validate_rect_bounds(diag_rect, width=width, height=height)
            enabled = True
            params["crop_rect"] = diag_rect.model_dump(mode="json")
            justification = "auto_crop selected explicitly; crop rectangle comes from cropdetect diagnostics."

    if choices.crop_mode == "custom_crop":
        if choices.custom_crop_margins is None:
            raise PreprocessingPlannerValidationError("custom_crop requires custom_crop_margins.")
        rect = _rect_from_margins(margins=choices.custom_crop_margins, width=width, height=height)
        _validate_rect_bounds(rect, width=width, height=height)
        enabled = True
        params["crop_rect"] = rect.model_dump(mode="json")
        justification = "custom_crop selected explicitly with validated manual margins."
        if width and height:
            src_ratio = width / height
            out_ratio = rect.width / rect.height
            if abs(src_ratio - out_ratio) / src_ratio > 0.08:
                warnings.append("Custom crop significantly changes content aspect ratio; downstream must avoid stretching.")

    return (
        PreprocessingOperation(
            operation_id="crop",
            kind="crop",
            enabled=enabled,
            recommended=recommended,
            confidence=diag_conf if diag_conf in {"high", "medium", "low", "unknown"} else "unknown",
            parameters=params,
            justification=justification,
            risk_notes=risk_notes,
        ),
        warnings,
    )


def _build_deinterlace_operation(
    *,
    diagnostics: dict[str, Any] | None,
    choices: PlannerUserChoices,
) -> tuple[PreprocessingOperation, list[str]]:
    warnings: list[str] = []
    classification = "indeterminate"
    confidence = "unknown"
    if diagnostics is not None and isinstance(diagnostics.get("idet"), dict):
        idet = diagnostics["idet"]
        classification = str(idet.get("classification") or "indeterminate")
        confidence = str(idet.get("confidence") or "unknown")

    recommended = classification == "interlaced" and confidence in {"high", "medium"}
    enabled = choices.deinterlace_mode == "enable"
    if enabled and classification == "progressive":
        warnings.append("Deinterlace was enabled manually although diagnostics indicate progressive content.")
    if enabled and classification in {"mixed", "indeterminate"}:
        warnings.append("Deinterlace was enabled manually with mixed/indeterminate cadence diagnostics.")

    params = {
        "mode": choices.deinterlace_mode,
        "ffmpeg_filter_hint": "bwdif=mode=send_frame:parity=auto:deint=all",
        "classification": classification,
    }
    justification = "Deinterlace is never auto-enabled; explicit user choice is required."
    return (
        PreprocessingOperation(
            operation_id="deinterlace",
            kind="deinterlace",
            enabled=enabled,
            recommended=recommended,
            confidence=confidence if confidence in {"high", "medium", "low", "unknown"} else "unknown",
            parameters=params,
            justification=justification,
            risk_notes=["Wrong deinterlace usage can soften detail or create artifacts."],
        ),
        warnings,
    )


def _build_noise_operation(
    *,
    diagnostics: dict[str, Any] | None,
    choices: PlannerUserChoices,
) -> tuple[PreprocessingOperation, list[str]]:
    warnings: list[str] = []
    likely_noise = "unknown"
    confidence = "unknown"
    if diagnostics is not None and isinstance(diagnostics.get("noise_artifacts"), dict):
        section = diagnostics["noise_artifacts"]
        likely_noise = str(section.get("likely_noise") or "unknown")
        confidence = str(section.get("confidence") or "unknown")
    recommended = likely_noise in {"high", "medium"} and confidence in {"high", "medium"}
    enabled = choices.denoise_mode == "enable"
    if enabled and likely_noise in {"low", "unknown"}:
        warnings.append("Denoise was enabled manually despite weak/unknown noise indicators.")
    return (
        PreprocessingOperation(
            operation_id="denoise",
            kind="denoise",
            enabled=enabled,
            recommended=recommended,
            confidence=confidence if confidence in {"high", "medium", "low", "unknown"} else "unknown",
            parameters={
                "mode": choices.denoise_mode,
                "ffmpeg_filter_hint": "hqdn3d=1.5:1.5:6:6",
                "strength": "light",
            },
            justification="Denoise is advisory only and requires explicit user selection.",
            risk_notes=["Over-denoising may erase texture and detail."],
        ),
        warnings,
    )


def _build_deblock_operation(
    *,
    diagnostics: dict[str, Any] | None,
    choices: PlannerUserChoices,
) -> tuple[PreprocessingOperation, list[str]]:
    warnings: list[str] = []
    likely_block = "unknown"
    confidence = "unknown"
    if diagnostics is not None and isinstance(diagnostics.get("noise_artifacts"), dict):
        section = diagnostics["noise_artifacts"]
        likely_block = str(section.get("likely_compression_artifacts") or "unknown")
        confidence = str(section.get("confidence") or "unknown")
    recommended = likely_block in {"high", "medium"} and confidence in {"high", "medium"}
    enabled = choices.deblock_mode == "enable"
    if enabled and likely_block in {"low", "unknown"}:
        warnings.append("Deblock was enabled manually despite weak/unknown compression-artifact indicators.")
    return (
        PreprocessingOperation(
            operation_id="deblock",
            kind="deblock",
            enabled=enabled,
            recommended=recommended,
            confidence=confidence if confidence in {"high", "medium", "low", "unknown"} else "unknown",
            parameters={
                "mode": choices.deblock_mode,
                "ffmpeg_filter_hint": "deblock=filter=strong:block=4",
                "strength": "light",
            },
            justification="Deblock is advisory only and requires explicit user selection.",
            risk_notes=["Aggressive deblocking may remove legitimate high-frequency detail."],
        ),
        warnings,
    )


def _estimate_impact(operations: list[PreprocessingOperation]) -> EstimatedImpact:
    enabled = [item for item in operations if item.enabled]
    if not enabled:
        return EstimatedImpact(
            cpu_cost="low",
            gpu_gain="unknown",
            quality_risk="low",
            note="No preprocessing operation is selected.",
        )
    cpu = "medium"
    risk = "medium"
    if any(item.kind in {"deinterlace", "denoise", "deblock"} for item in enabled):
        cpu = "high"
    if len(enabled) == 1 and enabled[0].kind == "crop":
        cpu = "low"
    if any(item.kind in {"denoise", "deblock"} for item in enabled):
        risk = "high"
    return EstimatedImpact(
        cpu_cost=cpu,
        gpu_gain="unknown",
        quality_risk=risk,
        note="Impact is qualitative only; no acceleration percentage is estimated.",
    )


def _rect_from_margins(*, margins: CropMargins, width: int | None, height: int | None) -> CropRect:
    if width is None or height is None:
        raise PreprocessingPlannerValidationError("custom_crop requires source width/height.")
    w = width - margins.left - margins.right
    h = height - margins.top - margins.bottom
    if w <= 0 or h <= 0:
        raise PreprocessingPlannerValidationError("custom_crop margins produce a non-positive crop rectangle.")
    return CropRect(x=margins.left, y=margins.top, width=w, height=h)


def _validate_rect_bounds(rect: CropRect, *, width: int | None, height: int | None) -> None:
    if width is None or height is None:
        raise PreprocessingPlannerValidationError("Crop validation requires source width/height.")
    if rect.x + rect.width > width or rect.y + rect.height > height:
        raise PreprocessingPlannerValidationError("Crop rectangle is out of source bounds.")
    if rect.width < 16 or rect.height < 16:
        raise PreprocessingPlannerValidationError("Crop rectangle is too small to be practical.")
