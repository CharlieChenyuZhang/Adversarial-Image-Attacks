"""Local browser UI for generating and evaluating adversarial candidates."""

from __future__ import annotations

import argparse
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


def generate_candidate(
    base: Image.Image,
    guide: Image.Image,
    strength: float,
    epsilon_levels: float,
    resize_mode: str,
) -> Tuple[Image.Image, str]:
    """Generate an image and a Markdown summary for the local UI."""

    if base is None or guide is None:
        raise ValueError("请先上传 base image 和 guide image。")
    if not isinstance(base, Image.Image) or not isinstance(guide, Image.Image):
        raise ValueError("请输入 PNG、JPEG、WEBP 等位图格式，暂不支持 SVG。")
    epsilon_value = float(epsilon_levels)
    if not 0.0 <= epsilon_value <= 255.0:
        raise ValueError("epsilon 必须在 0 到 255 个像素级别之间。")

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
        "### 生成完成\n"
        f"- 输出尺寸：`{candidate.width} × {candidate.height}`\n"
        f"- 最大通道变化：`{maximum}/255`\n"
        f"- 平均通道变化：`{mean:.2f}/255`\n"
        f"- 请求的 epsilon：`{epsilon_value:g}/255`\n\n"
        "候选图不等于已证明的攻击成功。"
        "请继续运行下方模型对比。"
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
    token_text = "不可用"
    if analysis.input_tokens is not None or analysis.output_tokens is not None:
        token_text = (
            f"输入 {analysis.input_tokens or 0}，"
            f"输出 {analysis.output_tokens or 0}"
        )
    return (
        f"### {title}\n"
        f"- 主要标签：**{_markdown_text(analysis.label)}**\n"
        f"- 置信度：`{analysis.confidence:.2f}`\n"
        f"- 匹配目标：`{'是' if analysis.target_match else '否'}`\n"
        f"- 目标判断：{_markdown_text(analysis.target_reason)}\n"
        f"- Token：{token_text}\n\n"
        f"{_markdown_text(analysis.description)}"
    )


def _verdict_markdown(
    comparison: VisionComparison, target_label: str, model: str
) -> str:
    target = target_label.strip()
    if target:
        if comparison.targeted_success:
            verdict = "定向攻击判定：成功信号"
            detail = "原图不匹配目标，而候选图匹配目标。"
        elif comparison.base.target_match:
            verdict = "定向攻击判定：无法成立"
            detail = (
                "原图已经匹配目标，"
                "不能把候选图匹配视为攻击造成。"
            )
        else:
            verdict = "定向攻击判定：未成功"
            detail = "候选图没有被模型判断为目标标签。"
    else:
        verdict = (
            "非定向变化信号：标签发生变化"
            if comparison.label_changed
            else "非定向变化信号：标签未变化"
        )
        detail = (
            "未提供目标标签，因此这里只能比较独立输出，"
            "不能证明定向攻击。"
        )

    return (
        "### 对比结论\n"
        f"**{verdict}**\n\n"
        f"{detail}\n\n"
        f"- 模型：`{model}`\n"
        f"- 原图标签：{_markdown_text(comparison.base.label)}\n"
        f"- 候选图标签：{_markdown_text(comparison.candidate.label)}\n"
        f"- 标签是否变化：`{'是' if comparison.label_changed else '否'}`\n\n"
        "单次大模型回答存在随机性。"
        "严谨实验应重复多次并预先定义成功标准。"
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
        raise ValueError("请先上传原图并生成候选图。")
    if not isinstance(base, Image.Image) or not isinstance(candidate, Image.Image):
        raise ValueError("模型评估只支持位图格式，暂不支持 SVG。")
    evaluator = OpenAIVisionEvaluator(api_key=api_key or None, model=DEFAULT_MODEL)
    comparison = evaluator.compare(
        _processed_base(base),
        candidate,
        prompt=prompt,
        target_label=target_label,
        reasoning_effort=reasoning_effort,
    )
    return (
        _analysis_markdown("原图分析", comparison.base),
        _analysis_markdown("候选图分析", comparison.candidate),
        _verdict_markdown(comparison, target_label, evaluator.model),
    )


def build_app():
    """Build the Gradio Blocks app, importing Gradio only when requested."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "缺少 UI 依赖。请使用 Python 3.10+ 运行："
            "python -m pip install -e '.[ui]'"
        ) from exc

    def generate_for_ui(*args):
        try:
            return generate_candidate(*args)
        except Exception as exc:
            expected = isinstance(
                exc, (TypeError, ValueError, VisionEvaluationError)
            )
            raise gr.Error(
                str(exc), print_exception=not expected
            ) from exc

    def evaluate_for_ui(*args):
        try:
            return evaluate_candidate(*args)
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
            "上传两张图片，生成有界候选图，"
            "然后用 GPT-5.6 Sol 独立比较原图和候选图。"
        )
        with gr.Row():
            base_image = gr.Image(
                label="Base image（希望保持视觉内容的原图）",
                type="pil",
                image_mode=None,
            )
            guide_image = gr.Image(
                label="Guide image（扰动方向或目标图）",
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
                label="Epsilon（像素级别，8 表示 8/255）",
            )
            resize_mode = gr.Dropdown(
                choices=["cover", "stretch"],
                value="cover",
                label="Guide resize mode",
            )
        generate_button = gr.Button("1. 生成候选图", variant="primary")
        with gr.Row():
            candidate_image = gr.Image(
                label="Adversarial candidate",
                type="pil",
                image_mode=None,
                format="png",
                interactive=False,
            )
            generation_summary = gr.Markdown()

        gr.Markdown("## OpenAI 视觉模型对比")
        gr.Markdown(
            f"固定模型：`{DEFAULT_MODEL}`。只有点击下方对比按钮时，"
            "原图和候选图才会发送到 OpenAI。"
        )
        prompt = gr.Textbox(
            label="评估问题",
            value=DEFAULT_PROMPT,
            lines=3,
        )
        with gr.Row():
            target_label = gr.Textbox(
                label="目标标签（可选，例如 dog）",
                placeholder="留空则只检测标签是否发生变化",
            )
            reasoning_effort = gr.Dropdown(
                choices=["low", "medium", "high"],
                value="medium",
                label="Reasoning effort",
            )
            api_key = gr.Textbox(
                label="OpenAI API Key（可选）",
                placeholder="留空则读取 OPENAI_API_KEY",
                type="password",
            )
        evaluate_button = gr.Button("2. 用 GPT-5.6 Sol 独立对比")
        with gr.Row():
            base_result = gr.Markdown()
            candidate_result = gr.Markdown()
        verdict = gr.Markdown()
        gr.Markdown(
            "隐私提示：API Key 不会写入仓库。"
            "图片输入会按照 OpenAI API 的数据与保留设置处理。"
            "请只测试你有权使用的图片和模型。"
        )

        generate_button.click(
            fn=generate_for_ui,
            inputs=[base_image, guide_image, strength, epsilon, resize_mode],
            outputs=[candidate_image, generation_summary],
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
            ],
            outputs=[base_result, candidate_result, verdict],
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
