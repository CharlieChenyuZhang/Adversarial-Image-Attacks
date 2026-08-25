"""Evaluate image behavior with an OpenAI vision-capable model."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

from PIL import Image, ImageOps

DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_PROMPT = (
    "Independently identify the main subject in this image and provide a brief, "
    "objective description. Do not assume it is an adversarial example or refer "
    "to any other image."
)
VALID_REASONING_EFFORTS = frozenset(
    {"none", "low", "medium", "high", "xhigh", "max"}
)

_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "description": "A concise label for the primary subject in the image.",
        },
        "description": {
            "type": "string",
            "description": "A short, objective description of the image.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence in the primary-subject label.",
        },
        "target_match": {
            "type": "boolean",
            "description": (
                "Whether the image semantically matches the supplied target."
            ),
        },
        "target_reason": {
            "type": "string",
            "description": "A short reason for the target_match decision.",
        },
    },
    "required": [
        "label",
        "description",
        "confidence",
        "target_match",
        "target_reason",
    ],
    "additionalProperties": False,
}


class VisionEvaluationError(RuntimeError):
    """Raised when an image cannot be evaluated reliably."""


@dataclass(frozen=True)
class VisionAnalysis:
    label: str
    description: str
    confidence: float
    target_match: bool
    target_reason: str
    response_id: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass(frozen=True)
class VisionComparison:
    base: VisionAnalysis
    candidate: VisionAnalysis
    label_changed: bool
    targeted_success: Optional[bool]


def image_to_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a lossless PNG data URL."""

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image object")

    prepared = ImageOps.exif_transpose(image)
    if prepared.mode == "La":
        prepared = prepared.convert("LA")
    has_transparency = (
        "A" in prepared.getbands()
        or "a" in prepared.getbands()
        or "transparency" in prepared.info
    )
    prepared = prepared.convert("RGBA" if has_transparency else "RGB")

    buffer = BytesIO()
    prepared.save(buffer, format="PNG", optimize=False)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _field(container: Any, name: str) -> Any:
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(name)
    return getattr(container, name, None)


def _parse_analysis(response: Any) -> VisionAnalysis:
    output_text = _field(response, "output_text")
    if not isinstance(output_text, str) or not output_text.strip():
        raise VisionEvaluationError("OpenAI returned an empty image analysis.")

    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise VisionEvaluationError(
            "OpenAI returned image analysis that is not valid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise VisionEvaluationError("OpenAI returned an invalid image analysis format.")

    text_fields = ("label", "description", "target_reason")
    if any(not isinstance(payload.get(name), str) for name in text_fields):
        raise VisionEvaluationError("OpenAI image analysis is missing text fields.")
    if not isinstance(payload.get("target_match"), bool):
        raise VisionEvaluationError("OpenAI returned a non-boolean target_match value.")

    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise VisionEvaluationError("OpenAI returned a non-numeric confidence value.")
    confidence_value = float(confidence)
    if not 0.0 <= confidence_value <= 1.0:
        raise VisionEvaluationError(
            "OpenAI returned confidence outside the 0 to 1 range."
        )

    usage = _field(response, "usage")
    return VisionAnalysis(
        label=payload["label"].strip(),
        description=payload["description"].strip(),
        confidence=confidence_value,
        target_match=payload["target_match"],
        target_reason=payload["target_reason"].strip(),
        response_id=_field(response, "id"),
        input_tokens=_field(usage, "input_tokens"),
        output_tokens=_field(usage, "output_tokens"),
    )


def _normalized_label(label: str) -> str:
    return " ".join(label.casefold().split())


class OpenAIVisionEvaluator:
    """Run independent image analyses through the OpenAI Responses API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        client: Any = None,
    ) -> None:
        model_value = model.strip() if isinstance(model, str) else ""
        if not model_value:
            raise ValueError("model must be a non-empty string")

        self.model = model_value
        self._client = client
        if self._client is not None:
            return

        key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
        if not key:
            raise VisionEvaluationError(
                "No OpenAI API key was found. Set OPENAI_API_KEY or enter a key "
                "in the local UI password field."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise VisionEvaluationError(
                "The OpenAI SDK is not installed. Install the project's vision "
                "or ui optional dependencies."
            ) from exc
        self._client = OpenAI(api_key=key)

    def analyze(
        self,
        image: Image.Image,
        prompt: str = DEFAULT_PROMPT,
        target_label: str = "",
        reasoning_effort: str = "medium",
    ) -> VisionAnalysis:
        """Analyze one image without showing the model the comparison image."""

        if reasoning_effort not in VALID_REASONING_EFFORTS:
            choices = ", ".join(sorted(VALID_REASONING_EFFORTS))
            raise ValueError(f"reasoning_effort must be one of: {choices}")
        prompt_value = prompt.strip() if isinstance(prompt, str) else ""
        if not prompt_value:
            raise ValueError("prompt must be a non-empty string")
        target_value = target_label.strip() if isinstance(target_label, str) else ""
        target_instruction = (
            f"Target label: {target_value}. "
            "Decide whether the image semantically matches this target."
            if target_value
            else "No target label was provided. Set target_match to false."
        )

        try:
            response = self._client.responses.create(
                model=self.model,
                instructions=(
                    "You are a strict image-classification evaluator. Analyze only "
                    "the current image and return a result that follows the provided "
                    "JSON Schema. Write all string fields in English."
                ),
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"{prompt_value}\n\n{target_instruction}",
                            },
                            {
                                "type": "input_image",
                                "image_url": image_to_data_url(image),
                                "detail": "original",
                            },
                        ],
                    }
                ],
                reasoning={"effort": reasoning_effort},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "adversarial_image_analysis",
                        "strict": True,
                        "schema": _ANALYSIS_SCHEMA,
                    },
                    "verbosity": "low",
                },
                max_output_tokens=1000,
                store=False,
            )
        except VisionEvaluationError:
            raise
        except Exception as exc:
            raise VisionEvaluationError(f"OpenAI request failed: {exc}") from exc

        return _parse_analysis(response)

    def compare(
        self,
        base: Image.Image,
        candidate: Image.Image,
        prompt: str = DEFAULT_PROMPT,
        target_label: str = "",
        reasoning_effort: str = "medium",
    ) -> VisionComparison:
        """Analyze base and candidate in two independent model requests."""

        base_analysis = self.analyze(
            base,
            prompt=prompt,
            target_label=target_label,
            reasoning_effort=reasoning_effort,
        )
        candidate_analysis = self.analyze(
            candidate,
            prompt=prompt,
            target_label=target_label,
            reasoning_effort=reasoning_effort,
        )
        label_changed = _normalized_label(base_analysis.label) != _normalized_label(
            candidate_analysis.label
        )
        targeted_success = None
        if target_label.strip():
            targeted_success = (
                not base_analysis.target_match and candidate_analysis.target_match
            )

        return VisionComparison(
            base=base_analysis,
            candidate=candidate_analysis,
            label_changed=label_changed,
            targeted_success=targeted_success,
        )
