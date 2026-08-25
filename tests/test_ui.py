import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

import adversarial_image_attacks.ui as ui_module
from adversarial_image_attacks.ui import (
    evaluate_candidate,
    generate_candidate,
    main,
)
from adversarial_image_attacks.vision import VisionAnalysis, VisionComparison


def analysis(
    label: str,
    *,
    target_match: bool,
    description: str = "Description.",
) -> VisionAnalysis:
    return VisionAnalysis(
        label=label,
        description=description,
        confidence=0.8,
        target_match=target_match,
        target_reason="Target reason.",
        response_id="resp_test",
        input_tokens=100,
        output_tokens=20,
    )


def test_generate_candidate_uses_pixel_level_epsilon() -> None:
    base = Image.new("RGB", (2, 2), (100, 100, 100))
    guide = Image.new("RGB", (2, 2), (200, 0, 100))

    candidate, summary = generate_candidate(base, guide, 1.0, 8, "cover")

    assert np.asarray(candidate).reshape(-1, 3).tolist() == [[108, 92, 100]] * 4
    assert "8/255" in summary
    assert "2 × 2" in summary


def test_generate_candidate_requires_both_images() -> None:
    image = Image.new("RGB", (1, 1))

    with pytest.raises(ValueError, match="base image"):
        generate_candidate(None, image, 1.0, 8, "cover")  # type: ignore[arg-type]


def test_candidate_fingerprint_tracks_every_generation_input() -> None:
    base = Image.new("RGB", (2, 2), "black")
    guide = Image.new("RGB", (2, 2), "white")
    fingerprint = ui_module._candidate_inputs_fingerprint(
        base, guide, 1.0, 8, "cover"
    )
    changed_base = base.copy()
    changed_base.putpixel((0, 0), (1, 0, 0))
    changed_guide = guide.copy()
    changed_guide.putpixel((0, 0), (254, 255, 255))

    changed_inputs = [
        (changed_base, guide, 1.0, 8, "cover"),
        (base, changed_guide, 1.0, 8, "cover"),
        (base, guide, 0.5, 8, "cover"),
        (base, guide, 1.0, 9, "cover"),
        (base, guide, 1.0, 8, "stretch"),
    ]

    for inputs in changed_inputs:
        assert ui_module._candidate_inputs_fingerprint(*inputs) != fingerprint


def test_candidate_fingerprint_tracks_palette_changes() -> None:
    base = Image.new("P", (1, 1), 0)
    guide = Image.new("RGB", (1, 1), "white")
    base.putpalette([0, 0, 0] + [0, 0, 0] * 255)
    fingerprint = ui_module._candidate_inputs_fingerprint(
        base, guide, 1.0, 8, "cover"
    )

    base.putpalette([255, 0, 0] + [0, 0, 0] * 255)

    assert (
        ui_module._candidate_inputs_fingerprint(
            base, guide, 1.0, 8, "cover"
        )
        != fingerprint
    )


def test_current_candidate_validation_rejects_changed_inputs() -> None:
    base = Image.new("RGB", (2, 2), "black")
    guide = Image.new("RGB", (2, 2), "white")
    fingerprint = ui_module._candidate_inputs_fingerprint(
        base, guide, 1.0, 8, "cover"
    )

    ui_module._require_current_candidate(
        fingerprint, base, guide, 1.0, 8, "cover"
    )

    with pytest.raises(ValueError, match="Generate a new candidate"):
        ui_module._require_current_candidate(
            fingerprint, base, guide, 1.0, 9, "cover"
        )


def test_current_candidate_validation_requires_generation() -> None:
    image = Image.new("RGB", (1, 1))

    with pytest.raises(ValueError, match="current settings first"):
        ui_module._require_current_candidate(
            "", image, image, 1.0, 8, "cover"
        )


def test_clear_candidate_state_removes_candidate_and_all_results() -> None:
    assert ui_module._clear_candidate_state() == (None, "", "", "", "", "")


def test_evaluate_candidate_formats_independent_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = VisionComparison(
        base=analysis("dog", target_match=False, description="A dog."),
        candidate=analysis("cat", target_match=True, description="A cat."),
        label_changed=True,
        targeted_success=True,
    )
    evaluator = SimpleNamespace(
        model="gpt-5.6-sol",
        compare=lambda *args, **kwargs: comparison,
    )
    constructor = lambda **kwargs: evaluator
    monkeypatch.setattr(ui_module, "OpenAIVisionEvaluator", constructor)

    base_result, candidate_result, verdict = evaluate_candidate(
        Image.new("RGB", (1, 1)),
        Image.new("RGB", (1, 1), "white"),
        "Classify this image.",
        "cat",
        "medium",
        "test-key",
    )

    assert "dog" in base_result
    assert "cat" in candidate_result
    assert "success signal" in verdict
    assert "gpt-5.6-sol" in verdict


def test_evaluate_candidate_escapes_untrusted_model_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malicious = "![tracking](https://example.invalid/pixel)<img src=x>"
    comparison = VisionComparison(
        base=analysis(malicious, target_match=False, description=malicious),
        candidate=analysis("safe", target_match=False, description="safe"),
        label_changed=True,
        targeted_success=None,
    )
    evaluator = SimpleNamespace(
        model="gpt-5.6-sol",
        compare=lambda *args, **kwargs: comparison,
    )
    monkeypatch.setattr(
        ui_module,
        "OpenAIVisionEvaluator",
        lambda **kwargs: evaluator,
    )

    base_result, _, verdict = evaluate_candidate(
        Image.new("RGB", (1, 1)),
        Image.new("RGB", (1, 1), "white"),
        "Classify this image.",
        "",
        "medium",
        "test-key",
    )

    assert "![tracking]" not in base_result
    assert "<img" not in base_result
    assert "https://" not in base_result
    assert "https://" not in verdict
    assert "&#33;&#91;tracking&#93;" in base_result


def test_build_app_when_gradio_is_installed() -> None:
    if importlib.util.find_spec("gradio") is None:
        pytest.skip("gradio is not installed")
    environment = os.environ.copy()
    environment["GRADIO_ANALYTICS_ENABLED"] = "False"
    project_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from adversarial_image_attacks.ui import build_app; "
                "app = build_app(); assert app is not None; "
                "deps = app.config['dependencies']; "
                "changes = [d for d in deps "
                "if any(t[1] == 'change' for t in d['targets'])]; "
                "assert len(changes) == 5; "
                "assert all(d['queue'] is False for d in changes); "
                "assert all(len(d['outputs']) == 6 for d in changes); "
                "assert len(deps[0]['outputs']) == 6; "
                "assert len(deps[1]['inputs']) == 11; "
                "app.close()"
            ),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_main_launches_only_on_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    launch = SimpleNamespace(calls=[])

    class FakeApp:
        def launch(self, **kwargs):
            launch.calls.append(kwargs)

    monkeypatch.setattr(ui_module, "build_app", lambda: FakeApp())

    assert main(["--port", "8123", "--no-browser"]) == 0
    assert launch.calls == [
        {
            "server_name": "127.0.0.1",
            "server_port": 8123,
            "inbrowser": False,
            "share": False,
            "max_file_size": "20mb",
        }
    ]


def test_main_rejects_invalid_port() -> None:
    with pytest.raises(SystemExit):
        main(["--port", "70000"])
