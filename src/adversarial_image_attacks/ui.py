"""Local browser UI for generating and evaluating adversarial candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .merge import merge_images
from .vision import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    OpenAIVisionEvaluator,
    VisionAnalysis,
    VisionComparison,
    VisionEvaluationError,
)


def _processed_base(base: Image.Image) -> Image.Image:
    return merge_images(base, base, strength=0.0, epsilon=1.0)


def _candidate_inputs_fingerprint(
    base: Image.Image,
    guide: Image.Image,
    strength: float,
    epsilon_levels: float,
    resize_mode: str,
) -> str:
    """Return a deterministic fingerprint for every candidate input."""

    if not isinstance(base, Image.Image) or not isinstance(guide, Image.Image):
        raise ValueError("Upload both a base image and a guide image.")

    digest = hashlib.sha256()
    settings = json.dumps(
        {
            "strength": float(strength),
            "epsilon_levels": float(epsilon_levels),
            "resize_mode": resize_mode,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(settings).to_bytes(8, "big"))
    digest.update(settings)

    for image in (base, guide):
        normalized = _processed_base(image)
        metadata = json.dumps(
            {
                "mode": normalized.mode,
                "size": normalized.size,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        pixels = normalized.tobytes()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(pixels).to_bytes(8, "big"))
        digest.update(pixels)

    return digest.hexdigest()


def _require_current_candidate(
    generated_fingerprint: str,
    base: Image.Image,
    guide: Image.Image,
    strength: float,
    epsilon_levels: float,
    resize_mode: str,
) -> None:
    """Reject a candidate generated from inputs that are no longer current."""

    if not generated_fingerprint:
        raise ValueError("Generate a candidate with the current settings first.")
    current_fingerprint = _candidate_inputs_fingerprint(
        base,
        guide,
        strength,
        epsilon_levels,
        resize_mode,
    )
    if not secrets.compare_digest(generated_fingerprint, current_fingerprint):
        raise ValueError(
            "The images or generation settings changed. "
            "Generate a new candidate before evaluation."
        )


def _clear_candidate_state() -> Tuple[None, str, str, str, str, str]:
    """Clear a stale candidate, its fingerprint, and prior model results."""

    return None, "", "", "", "", ""


def generate_candidate(
    base: Image.Image,
    guide: Image.Image,
    strength: float,
    epsilon_levels: float,
    resize_mode: str,
) -> Tuple[Image.Image, str]:
    """Generate an image and a Markdown summary for the local UI."""

    if base is None or guide is None:
        raise ValueError("Upload both a base image and a guide image.")
    if not isinstance(base, Image.Image) or not isinstance(guide, Image.Image):
        raise ValueError(
            "Use a bitmap image such as PNG, JPEG, or WebP. "
            "SVG is not supported."
        )
    epsilon_value = float(epsilon_levels)
    if not 0.0 <= epsilon_value <= 255.0:
        raise ValueError("epsilon must be between 0 and 255 pixel levels.")

    candidate = merge_images(
        base,
        guide,
        strength=float(strength),
        epsilon=epsilon_value / 255.0,
        resize_mode=resize_mode,
    )
    reference = _processed_base(base)
    candidate_pixels = np.asarray(candidate, dtype=np.int16)
    reference_pixels = np.asarray(reference, dtype=np.int16)
    delta = np.abs(candidate_pixels - reference_pixels)
    maximum = int(delta.max(initial=0))
    mean = float(delta.mean()) if delta.size else 0.0
    summary = (
        "### Candidate ready\n"
        f"- Output size: `{candidate.width} × {candidate.height}`\n"
        f"- Maximum channel change: `{maximum}/255`\n"
        f"- Mean channel change: `{mean:.2f}/255`\n"
        f"- Requested epsilon: `{epsilon_value:g}/255`\n\n"
        "This is only a candidate. Run the model comparison below to "
        "measure whether the label changed."
    )
    return candidate, summary


def _markdown_text(value: str) -> str:
    """Render untrusted model text without activating Markdown syntax."""

    rendered = []
    for character in value:
        if character.isascii() and not (
            character.isalnum() or character.isspace()
        ):
            rendered.append(f"&#{ord(character)};")
        elif character.isprintable() or character in "\r\n\t":
            rendered.append(character)
        else:
            rendered.append(" ")
    return "".join(rendered)


def _analysis_markdown(title: str, analysis: VisionAnalysis) -> str:
    token_text = "Unavailable"
    if analysis.input_tokens is not None or analysis.output_tokens is not None:
        token_text = (
            f"input {analysis.input_tokens or 0}, "
            f"output {analysis.output_tokens or 0}"
        )
    return (
        f"### {title}\n"
        f"- Primary label: **{_markdown_text(analysis.label)}**\n"
        f"- Confidence: `{analysis.confidence:.2f}`\n"
        f"- Matches target: `{'Yes' if analysis.target_match else 'No'}`\n"
        f"- Target reasoning: {_markdown_text(analysis.target_reason)}\n"
        f"- Tokens: {token_text}\n\n"
        f"{_markdown_text(analysis.description)}"
    )


def _verdict_markdown(
    comparison: VisionComparison, target_label: str, model: str
) -> str:
    target = target_label.strip()
    if target:
        if comparison.targeted_success:
            verdict = "Targeted result: success signal"
            detail = "The base did not match the target, but the candidate did."
        elif comparison.base.target_match:
            verdict = "Targeted result: inconclusive"
            detail = (
                "The base already matched the target, so a candidate match "
                "cannot be attributed to the perturbation."
            )
        else:
            verdict = "Targeted result: no success"
            detail = "The model did not match the candidate to the target."
    else:
        verdict = (
            "Untargeted signal: label changed"
            if comparison.label_changed
            else "Untargeted signal: label unchanged"
        )
        detail = (
            "No target label was provided, so this comparison cannot "
            "establish a targeted attack."
        )

    return (
        "### Comparison\n"
        f"**{verdict}**\n\n"
        f"{detail}\n\n"
        f"- Model: `{model}`\n"
        f"- Base label: {_markdown_text(comparison.base.label)}\n"
        f"- Candidate label: {_markdown_text(comparison.candidate.label)}\n"
        f"- Label changed: `{'Yes' if comparison.label_changed else 'No'}`\n\n"
        "A single model response is stochastic. Repeat the experiment and "
        "define the success rule before reviewing the results."
    )


def evaluate_candidate(
    base: Image.Image,
    candidate: Image.Image,
    prompt: str,
    target_label: str,
    reasoning_effort: str,
    api_key: str = "",
) -> Tuple[str, str, str]:
    """Evaluate base and candidate independently with GPT-5.6 Sol."""

    if base is None or candidate is None:
        raise ValueError("Upload a base image and generate a candidate first.")
    if not isinstance(base, Image.Image) or not isinstance(candidate, Image.Image):
        raise ValueError("Model evaluation supports bitmap images, not SVG.")
    evaluator = OpenAIVisionEvaluator(api_key=api_key or None, model=DEFAULT_MODEL)
    comparison = evaluator.compare(
        _processed_base(base),
        candidate,
        prompt=prompt,
        target_label=target_label,
        reasoning_effort=reasoning_effort,
    )
    return (
        _analysis_markdown("Base analysis", comparison.base),
        _analysis_markdown("Candidate analysis", comparison.candidate),
        _verdict_markdown(comparison, target_label, evaluator.model),
    )


def build_app():
    """Build the Gradio Blocks app, importing Gradio only when requested."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "UI dependencies are missing. With Python 3.10+, run: "
            "python -m pip install -e '.[ui]'"
        ) from exc

    def generate_for_ui(*args):
        try:
            candidate, summary = generate_candidate(*args)
            fingerprint = _candidate_inputs_fingerprint(*args)
            return candidate, summary, fingerprint, "", "", ""
        except Exception as exc:
            expected = isinstance(
                exc, (TypeError, ValueError, VisionEvaluationError)
            )
            raise gr.Error(
                str(exc), print_exception=not expected
            ) from exc

    def evaluate_for_ui(
        base,
        candidate,
        evaluation_prompt,
        evaluation_target,
        effort,
        key,
        generated_fingerprint,
        guide,
        candidate_strength,
        candidate_epsilon,
        candidate_resize_mode,
    ):
        try:
            _require_current_candidate(
                generated_fingerprint,
                base,
                guide,
                candidate_strength,
                candidate_epsilon,
                candidate_resize_mode,
            )
            return evaluate_candidate(
                base,
                candidate,
                evaluation_prompt,
                evaluation_target,
                effort,
                key,
            )
        except Exception as exc:
            expected = isinstance(
                exc, (TypeError, ValueError, VisionEvaluationError)
            )
            raise gr.Error(
                str(exc), print_exception=not expected
            ) from exc

    with gr.Blocks(
        title="Adversarial Image Lab", analytics_enabled=False
    ) as app:
        gr.Markdown(
            "# Adversarial Image Lab\n"
            "Upload a base and guide image, create a bounded candidate, "
            "then compare the base and candidate independently with "
            "GPT-5.6 Sol."
        )
        with gr.Row():
            base_image = gr.Image(
                label="Base image (content to preserve)",
                type="pil",
                image_mode=None,
            )
            guide_image = gr.Image(
                label="Guide image (perturbation direction)",
                type="pil",
                image_mode=None,
            )
        with gr.Row():
            strength = gr.Slider(
                minimum=0,
                maximum=1,
                value=1,
                step=0.05,
                label="Strength",
            )
            epsilon = gr.Slider(
                minimum=0,
                maximum=64,
                value=8,
                step=1,
                label="Epsilon (pixel levels; 8 means 8/255)",
            )
            resize_mode = gr.Dropdown(
                choices=["cover", "stretch"],
                value="cover",
                label="Guide resize mode",
            )
        generate_button = gr.Button("1. Generate candidate", variant="primary")
        with gr.Row():
            candidate_image = gr.Image(
                label="Adversarial candidate",
                type="pil",
                image_mode=None,
                format="png",
                interactive=False,
            )
            generation_summary = gr.Markdown()

        gr.Markdown("## OpenAI vision comparison")
        gr.Markdown(
            f"Fixed model: `{DEFAULT_MODEL}`. The images are sent to OpenAI "
            "only when you click the comparison button."
        )
        prompt = gr.Textbox(
            label="Evaluation prompt",
            value=DEFAULT_PROMPT,
            lines=3,
        )
        with gr.Row():
            target_label = gr.Textbox(
                label="Target label (optional, for example dog)",
                placeholder="Leave blank to check only whether the label changes",
            )
            reasoning_effort = gr.Dropdown(
                choices=["low", "medium", "high"],
                value="medium",
                label="Reasoning effort",
            )
            api_key = gr.Textbox(
                label="OpenAI API key (optional)",
                placeholder="Leave blank to use OPENAI_API_KEY",
                type="password",
            )
        evaluate_button = gr.Button("2. Compare with GPT-5.6 Sol")
        with gr.Row():
            base_result = gr.Markdown()
            candidate_result = gr.Markdown()
        verdict = gr.Markdown()
        candidate_fingerprint = gr.State("")
        gr.Markdown(
            "Privacy: the API key is not written to the repository. Image "
            "inputs follow your OpenAI API data and retention settings. "
            "Test only images and models you are authorized to use."
        )

        generate_button.click(
            fn=generate_for_ui,
            inputs=[base_image, guide_image, strength, epsilon, resize_mode],
            outputs=[
                candidate_image,
                generation_summary,
                candidate_fingerprint,
                base_result,
                candidate_result,
                verdict,
            ],
        )
        evaluate_button.click(
            fn=evaluate_for_ui,
            inputs=[
                base_image,
                candidate_image,
                prompt,
                target_label,
                reasoning_effort,
                api_key,
                candidate_fingerprint,
                guide_image,
                strength,
                epsilon,
                resize_mode,
            ],
            outputs=[base_result, candidate_result, verdict],
        )

        stale_outputs = [
            candidate_image,
            generation_summary,
            candidate_fingerprint,
            base_result,
            candidate_result,
            verdict,
        ]
        for candidate_input in (
            base_image,
            guide_image,
            strength,
            epsilon,
            resize_mode,
        ):
            candidate_input.change(
                fn=_clear_candidate_state,
                inputs=None,
                outputs=stale_outputs,
                queue=False,
            )

    return app


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="advmerge-ui",
        description="Launch the local Adversarial Image Lab UI.",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open a browser tab automatically",
    )
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.port <= 65535:
        parser.error("port must be between 1 and 65535")

    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=arguments.port,
        inbrowser=not arguments.no_browser,
        share=False,
        max_file_size="20mb",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
