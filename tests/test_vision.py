import base64
from io import BytesIO
import json
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
import pytest

from adversarial_image_attacks.vision import (
    DEFAULT_MODEL,
    OpenAIVisionEvaluator,
    VisionAnalysis,
    VisionComparison,
    VisionEvaluationError,
    image_to_data_url,
)


def response_with(
    *,
    label: str,
    description: str,
    confidence: float,
    target_match: bool,
    target_reason: str,
    response_id: str,
    input_tokens: int,
    output_tokens: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=response_id,
        output_text=json.dumps(
            {
                "label": label,
                "description": description,
                "confidence": confidence,
                "target_match": target_match,
                "target_reason": target_reason,
            }
        ),
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def image_item_from_request(request: dict[str, object]) -> dict[str, object]:
    messages = request["input"]
    assert isinstance(messages, list)
    assert len(messages) == 1
    message = messages[0]
    assert isinstance(message, dict)
    assert message["role"] == "user"
    content = message["content"]
    assert isinstance(content, list)
    image_items = [
        item
        for item in content
        if isinstance(item, dict) and item.get("type") == "input_image"
    ]
    assert len(image_items) == 1
    return image_items[0]


def text_from_request(request: dict[str, object]) -> str:
    messages = request["input"]
    assert isinstance(messages, list)
    message = messages[0]
    assert isinstance(message, dict)
    content = message["content"]
    assert isinstance(content, list)
    text_items = [
        item["text"]
        for item in content
        if isinstance(item, dict) and item.get("type") == "input_text"
    ]
    assert text_items
    assert all(isinstance(text, str) for text in text_items)
    return "\n".join(text_items)


def test_image_to_data_url_encodes_a_lossless_png() -> None:
    image = Image.new("RGBA", (2, 1))
    image.putdata([(1, 2, 3, 4), (250, 251, 252, 253)])

    data_url = image_to_data_url(image)

    prefix, encoded = data_url.split(",", 1)
    assert prefix == "data:image/png;base64"
    with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as decoded:
        assert decoded.format == "PNG"
        assert decoded.mode == "RGBA"
        assert decoded.size == image.size
        np.testing.assert_array_equal(np.asarray(decoded), np.asarray(image))


def test_image_to_data_url_preserves_palette_transparency() -> None:
    image = Image.new("P", (1, 1), 0)
    image.putpalette([10, 20, 30] + [0, 0, 0] * 255)
    image.info["transparency"] = 0

    _, encoded = image_to_data_url(image).split(",", 1)

    with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as decoded:
        assert decoded.mode == "RGBA"
        assert tuple(np.asarray(decoded)[0, 0]) == (10, 20, 30, 0)


def test_analyze_uses_gpt_5_6_sol_and_parses_structured_response() -> None:
    client = MagicMock()
    client.responses.create.return_value = response_with(
        label="tabby cat",
        description="A cat sitting on a woven mat.",
        confidence=0.875,
        target_match=True,
        target_reason="The dominant subject is a cat.",
        response_id="resp_candidate",
        input_tokens=321,
        output_tokens=47,
    )
    evaluator = OpenAIVisionEvaluator(client=client)
    image = Image.new("RGB", (3, 2), (12, 34, 56))

    analysis = evaluator.analyze(
        image,
        "Identify the primary object.",
        target_label="cat",
        reasoning_effort="high",
    )

    assert DEFAULT_MODEL == "gpt-5.6-sol"
    assert analysis == VisionAnalysis(
        label="tabby cat",
        description="A cat sitting on a woven mat.",
        confidence=0.875,
        target_match=True,
        target_reason="The dominant subject is a cat.",
        response_id="resp_candidate",
        input_tokens=321,
        output_tokens=47,
    )
    assert isinstance(analysis.confidence, float)

    client.responses.create.assert_called_once()
    request = client.responses.create.call_args.kwargs
    assert request["model"] == "gpt-5.6-sol"
    assert request["store"] is False
    assert request["reasoning"] == {"effort": "high"}
    image_item = image_item_from_request(request)
    assert image_item["detail"] == "original"
    assert str(image_item["image_url"]).startswith("data:image/png;base64,")
    request_text = text_from_request(request)
    assert "Identify the primary object." in request_text
    assert "cat" in request_text

    text_config = request["text"]
    assert isinstance(text_config, dict)
    output_format = text_config["format"]
    assert isinstance(output_format, dict)
    assert output_format["type"] == "json_schema"
    assert output_format["strict"] is True
    schema = output_format["schema"]
    assert isinstance(schema, dict)
    assert set(schema["required"]) == {
        "label",
        "description",
        "confidence",
        "target_match",
        "target_reason",
    }


@pytest.mark.parametrize(
    (
        "target_label",
        "base_matches_target",
        "candidate_matches_target",
        "expected_targeted_success",
    ),
    [
        ("cat", False, True, True),
        ("cat", False, False, False),
        ("cat", True, True, False),
        ("", False, False, None),
    ],
)
def test_compare_uses_two_independent_calls_and_reports_attack_verdicts(
    target_label: str,
    base_matches_target: bool,
    candidate_matches_target: bool,
    expected_targeted_success: Optional[bool],
) -> None:
    client = MagicMock()
    client.responses.create.side_effect = [
        response_with(
            label="dog",
            description="A dog outdoors.",
            confidence=0.98,
            target_match=base_matches_target,
            target_reason="Base target verdict.",
            response_id="resp_base",
            input_tokens=100,
            output_tokens=20,
        ),
        response_with(
            label="cat" if candidate_matches_target else "bird",
            description="The candidate image result.",
            confidence=0.76,
            target_match=candidate_matches_target,
            target_reason="Candidate target verdict.",
            response_id="resp_candidate",
            input_tokens=101,
            output_tokens=21,
        ),
    ]
    evaluator = OpenAIVisionEvaluator(client=client)
    base = Image.new("RGB", (2, 2), (255, 0, 0))
    candidate = Image.new("RGB", (2, 2), (0, 0, 255))

    comparison = evaluator.compare(
        base,
        candidate,
        prompt="Classify the main subject.",
        target_label=target_label,
        reasoning_effort="low",
    )

    assert isinstance(comparison, VisionComparison)
    assert comparison.base.label == "dog"
    assert comparison.base.response_id == "resp_base"
    assert comparison.candidate.response_id == "resp_candidate"
    assert comparison.label_changed is True
    assert comparison.targeted_success is expected_targeted_success

    assert client.responses.create.call_count == 2
    first_request = client.responses.create.call_args_list[0].kwargs
    second_request = client.responses.create.call_args_list[1].kwargs
    for request in (first_request, second_request):
        assert request["model"] == DEFAULT_MODEL
        assert request["store"] is False
        assert request["reasoning"] == {"effort": "low"}
        assert image_item_from_request(request)["detail"] == "original"
    assert (
        image_item_from_request(first_request)["image_url"]
        != image_item_from_request(second_request)["image_url"]
    )


def test_compare_reports_no_untargeted_success_when_label_is_unchanged() -> None:
    client = MagicMock()
    client.responses.create.side_effect = [
        response_with(
            label="dog",
            description="Base.",
            confidence=0.9,
            target_match=False,
            target_reason="No target requested.",
            response_id="resp_base",
            input_tokens=10,
            output_tokens=2,
        ),
        response_with(
            label="dog",
            description="Candidate.",
            confidence=0.8,
            target_match=False,
            target_reason="No target requested.",
            response_id="resp_candidate",
            input_tokens=11,
            output_tokens=3,
        ),
    ]

    comparison = OpenAIVisionEvaluator(client=client).compare(
        Image.new("RGB", (1, 1)),
        Image.new("RGB", (1, 1), "white"),
        prompt="Classify this image.",
    )

    assert comparison.label_changed is False
    assert comparison.targeted_success is None


def test_constructor_requires_an_api_key_without_an_injected_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(VisionEvaluationError, match="OPENAI_API_KEY"):
        OpenAIVisionEvaluator()


def test_analyze_rejects_malformed_json() -> None:
    client = MagicMock()
    client.responses.create.return_value = SimpleNamespace(
        id="resp_bad",
        output_text="this is not JSON",
        usage=SimpleNamespace(input_tokens=10, output_tokens=2),
    )

    with pytest.raises(VisionEvaluationError, match="JSON"):
        OpenAIVisionEvaluator(client=client).analyze(
            Image.new("RGB", (1, 1)),
            prompt="Classify this image.",
        )
