#!/usr/bin/env python3
"""Create a quantization-aware targeted PGD example for ResNet-18."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps


MODEL_NAME = "torchvision.models.resnet18"
WEIGHTS_NAME = "ResNet18_Weights.IMAGENET1K_V1"
WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


def _patch_old_torch_pytree(torch: Any) -> None:
    """Bridge a narrow compatibility gap in older PyTorch installations."""

    pytree = torch.utils._pytree
    if hasattr(pytree, "register_pytree_node"):
        return

    def register_pytree_node(
        node_type: object,
        flatten_fn: object,
        unflatten_fn: object,
        **kwargs: object,
    ) -> object:
        supported = {
            key: value
            for key, value in kwargs.items()
            if key in {"to_dumpable_context", "from_dumpable_context"}
        }
        return pytree._register_pytree_node(
            node_type, flatten_fn, unflatten_fn, **supported
        )

    pytree.register_pytree_node = register_pytree_node


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the attack dependencies with pip install -e '.[attack]'.") from exc

    _patch_old_torch_pytree(torch)
    try:
        import torchvision
        from torchvision import models
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as transform_functional
    except ImportError as exc:
        raise RuntimeError("Install the attack dependencies with pip install -e '.[attack]'.") from exc

    return torch, torchvision, models, (InterpolationMode, transform_functional)


def _parse_levels(value: str) -> list[int]:
    try:
        levels = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("epsilon levels must be comma-separated integers") from exc
    if not levels or levels != sorted(set(levels)) or levels[0] < 1 or levels[-1] > 16:
        raise argparse.ArgumentTypeError(
            "epsilon levels must be unique ascending integers between 1 and 16"
        )
    return levels


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a targeted PGD attack against torchvision ResNet-18 V1."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("target", help="Exact ImageNet-1K target class name")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--epsilon-levels", type=_parse_levels, default=[1, 2, 4, 8])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional local resnet18-f37072fd.pth checkpoint",
    )
    return parser


def _select_device(torch: Any, requested: str) -> Any:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(torch: Any, models: Any, checkpoint: Path | None, device: Any) -> tuple[Any, Any, Path]:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    if checkpoint is None:
        model = models.resnet18(weights=weights)
        checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    else:
        checkpoint_path = checkpoint.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise ValueError(f"checkpoint not found: {checkpoint_path}")
        if _sha256(checkpoint_path) != WEIGHTS_SHA256:
            raise ValueError(
                "checkpoint hash does not match official ResNet-18 ImageNet-1K V1 weights"
            )
        model = models.resnet18(weights=None)
        model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )

    if _sha256(checkpoint_path) != WEIGHTS_SHA256:
        raise RuntimeError(
            "downloaded checkpoint hash does not match official ResNet-18 weights"
        )

    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, weights, checkpoint_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _canonical_pixels(path: Path, weights: Any, transform_runtime: Any) -> np.ndarray:
    interpolation_mode, functional = transform_runtime
    settings = weights.transforms()
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = functional.resize(
            image,
            list(settings.resize_size),
            interpolation=interpolation_mode.BILINEAR,
            antialias=True,
        )
        image = functional.center_crop(image, list(settings.crop_size))
        image.load()
    return np.asarray(image, dtype=np.uint8).copy()


def _to_tensor(torch: Any, pixels: np.ndarray, device: Any) -> Any:
    return (
        torch.from_numpy(pixels)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div(255.0)
    )


def _normalize(torch: Any, image: Any, weights: Any) -> Any:
    settings = weights.transforms()
    mean = torch.tensor(settings.mean, device=image.device).view(1, 3, 1, 1)
    std = torch.tensor(settings.std, device=image.device).view(1, 3, 1, 1)
    return (image - mean) / std


def _quantize_ste(torch: Any, image: Any) -> Any:
    quantized = torch.round(image * 255.0).clamp(0, 255).div(255.0)
    return image + (quantized - image).detach()


def _top5(torch: Any, model: Any, image: Any, weights: Any) -> tuple[Any, list[dict[str, Any]]]:
    with torch.inference_mode():
        logits = model(_normalize(torch, image, weights))[0]
        probabilities = logits.softmax(dim=0)
        values, indices = probabilities.topk(5)
    categories = weights.meta["categories"]
    rows = [
        {
            "index": int(index),
            "label": categories[int(index)],
            "probability": float(value),
        }
        for value, index in zip(values.cpu().tolist(), indices.cpu().tolist())
    ]
    return logits, rows


def _target_margin(torch: Any, logits: Any, target_index: int) -> float:
    target_logit = logits[target_index]
    other_logits = torch.cat((logits[:target_index], logits[target_index + 1 :]))
    return float((target_logit - other_logits.max()).detach().cpu())


def _attack_once(
    torch: Any,
    model: Any,
    clean: Any,
    target_index: int,
    weights: Any,
    *,
    epsilon_levels: int,
    steps: int,
    restarts: int,
    seed: int,
) -> tuple[Any, float]:
    import torch.nn.functional as functional

    epsilon = epsilon_levels / 255.0
    step_size = min(2, epsilon_levels) / 255.0
    target = torch.tensor([target_index], device=clean.device)
    random_generator = torch.Generator(device="cpu").manual_seed(seed + epsilon_levels)
    best_image = clean.detach().clone()
    clean_logits, _ = _top5(torch, model, clean, weights)
    best_margin = _target_margin(torch, clean_logits, target_index)

    for restart in range(restarts):
        if restart == 0:
            adversarial = clean.detach().clone()
        else:
            delta_levels = torch.randint(
                -epsilon_levels,
                epsilon_levels + 1,
                clean.shape,
                generator=random_generator,
                device="cpu",
            )
            adversarial = (
                clean + delta_levels.to(clean.device, dtype=clean.dtype) / 255.0
            ).clamp(0.0, 1.0)

        with torch.inference_mode():
            initial = torch.round(adversarial * 255.0).clamp(0, 255).div(255.0)
            initial_logits = model(_normalize(torch, initial, weights))[0]
        initial_margin = _target_margin(torch, initial_logits, target_index)
        if initial_margin > best_margin:
            best_margin = initial_margin
            best_image = initial.detach().clone()

        for _ in range(steps):
            adversarial = adversarial.detach().requires_grad_(True)
            quantized = _quantize_ste(torch, adversarial)
            logits = model(_normalize(torch, quantized, weights))
            loss = functional.cross_entropy(logits, target)
            gradient = torch.autograd.grad(loss, adversarial)[0]
            adversarial = adversarial.detach() - step_size * gradient.sign()
            adversarial = torch.maximum(
                torch.minimum(adversarial, clean + epsilon), clean - epsilon
            ).clamp(0.0, 1.0)

            with torch.inference_mode():
                candidate = torch.round(adversarial * 255.0).clamp(0, 255).div(255.0)
                candidate_logits = model(_normalize(torch, candidate, weights))[0]
            margin = _target_margin(torch, candidate_logits, target_index)
            if margin > best_margin:
                best_margin = margin
                best_image = candidate.detach().clone()

    return best_image, best_margin


def _save_png(torch: Any, image: Any, path: Path) -> None:
    pixels = (
        torch.round(image.detach().cpu()[0].permute(1, 2, 0) * 255.0)
        .clamp(0, 255)
        .to(torch.uint8)
        .numpy()
    )
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG", optimize=False)


def _load_png_tensor(torch: Any, path: Path, device: Any) -> tuple[np.ndarray, Any]:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return pixels, _to_tensor(torch, pixels, device)


def _metrics(clean: np.ndarray, adversarial: np.ndarray) -> dict[str, Any]:
    signed = adversarial.astype(np.int16) - clean.astype(np.int16)
    absolute = np.abs(signed)
    mse = float(np.mean(np.square(signed.astype(np.float64))))
    psnr = None if mse == 0 else 10.0 * math.log10((255.0**2) / mse)
    result: dict[str, Any] = {
        "linf_levels": int(absolute.max(initial=0)),
        "linf_normalized": float(absolute.max(initial=0) / 255.0),
        "mean_absolute_delta_levels": float(absolute.mean()),
        "changed_channels": int(np.count_nonzero(absolute)),
        "total_channels": int(absolute.size),
        "psnr_db": psnr,
    }
    try:
        from skimage.metrics import structural_similarity

        result["ssim"] = float(
            structural_similarity(clean, adversarial, channel_axis=2, data_range=255)
        )
    except (ImportError, TypeError):
        result["ssim"] = None
    return result


def _save_visualization(
    clean: np.ndarray,
    output: np.ndarray,
    path: Path,
    *,
    middle_label: str,
) -> None:
    difference = output.astype(np.int16) - clean.astype(np.int16)
    amplified = np.clip(128 + difference * 16, 0, 255).astype(np.uint8)
    panels = [Image.fromarray(clean), Image.fromarray(output), Image.fromarray(amplified)]
    labels = ["Clean", middle_label, "Perturbation x16"]
    width = sum(panel.width for panel in panels)
    canvas = Image.new("RGB", (width, panels[0].height + 28), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for panel, label in zip(panels, labels):
        canvas.paste(panel, (x, 28))
        draw.text((x + 8, 8), label, fill="black")
        x += panel.width
    canvas.save(path, format="PNG", optimize=False)


def main() -> int:
    args = _parser().parse_args()
    if not args.input.is_file():
        raise SystemExit(f"input image not found: {args.input}")
    if args.steps < 1 or args.restarts < 1:
        raise SystemExit("steps and restarts must be positive")

    torch, torchvision, models, transform_runtime = _load_runtime()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _select_device(torch, args.device)
    model, weights, checkpoint = _load_model(torch, models, args.checkpoint, device)
    categories = weights.meta["categories"]
    if args.target not in categories:
        raise SystemExit(f"target must exactly match an ImageNet class: {args.target!r}")
    target_index = categories.index(args.target)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_pixels = _canonical_pixels(args.input, weights, transform_runtime)
    clean_path = args.output_dir / "clean.png"
    Image.fromarray(clean_pixels, mode="RGB").save(clean_path, format="PNG", optimize=False)
    clean_pixels, clean = _load_png_tensor(torch, clean_path, device)
    clean_logits, clean_top5 = _top5(torch, model, clean, weights)
    if clean_top5[0]["index"] == target_index:
        raise SystemExit("clean input is already classified as the target class")

    attempts: list[dict[str, Any]] = []
    selected_image = clean
    selected_epsilon = 0
    selected_margin = _target_margin(torch, clean_logits, target_index)
    found_success = False
    for epsilon_levels in args.epsilon_levels:
        candidate, optimization_margin = _attack_once(
            torch,
            model,
            clean,
            target_index,
            weights,
            epsilon_levels=epsilon_levels,
            steps=args.steps,
            restarts=args.restarts,
            seed=args.seed,
        )
        candidate_logits, candidate_top5 = _top5(torch, model, candidate, weights)
        success = candidate_top5[0]["index"] == target_index
        attempts.append(
            {
                "epsilon_levels": epsilon_levels,
                "success_before_png_roundtrip": success,
                "target_margin": _target_margin(torch, candidate_logits, target_index),
                "optimization_margin": optimization_margin,
                "top1": candidate_top5[0],
            }
        )
        candidate_margin = _target_margin(torch, candidate_logits, target_index)
        if candidate_margin > selected_margin:
            selected_image = candidate
            selected_epsilon = epsilon_levels
            selected_margin = candidate_margin
        if success:
            selected_image = candidate
            selected_epsilon = epsilon_levels
            found_success = True
            break

    role = "adversarial" if found_success else "candidate"
    output_path = args.output_dir / f"{role}-eps-{selected_epsilon}.png"
    _save_png(torch, selected_image, output_path)
    output_pixels, reloaded = _load_png_tensor(torch, output_path, device)
    output_logits, output_top5 = _top5(torch, model, reloaded, weights)
    roundtrip_margin = _target_margin(torch, output_logits, target_index)
    success = output_top5[0]["index"] == target_index
    pixel_metrics = _metrics(clean_pixels, output_pixels)
    if pixel_metrics["linf_levels"] > selected_epsilon:
        raise RuntimeError("saved PNG exceeds the requested pixel budget")

    visualization_path = args.output_dir / "comparison.png"
    _save_visualization(
        clean_pixels,
        output_pixels,
        visualization_path,
        middle_label=role.title(),
    )
    result = {
        "success_after_png_roundtrip": success,
        "output_role": role,
        "files": {
            "input_source": {
                "path": _portable_path(args.input),
                "sha256": _sha256(args.input),
            },
            "clean_png": {
                "path": _portable_path(clean_path),
                "sha256": _sha256(clean_path),
            },
            "output_png": {
                "path": _portable_path(output_path),
                "sha256": _sha256(output_path),
            },
            "comparison_png": {
                "path": _portable_path(visualization_path),
                "sha256": _sha256(visualization_path),
            },
        },
        "model": MODEL_NAME,
        "weights": WEIGHTS_NAME,
        "weights_sha256": _sha256(checkpoint),
        "target": {"index": target_index, "label": args.target},
        "clean_top5": clean_top5,
        "output_top5": output_top5,
        "target_probability_clean": float(clean_logits.softmax(dim=0)[target_index].detach().cpu()),
        "target_probability_output": float(
            output_logits.softmax(dim=0)[target_index].detach().cpu()
        ),
        "target_margin_after_png_roundtrip": roundtrip_margin,
        "selected_epsilon_levels": selected_epsilon,
        "requested_epsilon_levels": args.epsilon_levels,
        "tested_epsilon_levels": [attempt["epsilon_levels"] for attempt in attempts],
        "steps": args.steps,
        "restarts": args.restarts,
        "seed": args.seed,
        "device": str(device),
        "preprocessing": "RGB, resize short side to 256, center crop to 224, uint8 PNG, normalize only during inference",
        "pixel_metrics_after_png_roundtrip": pixel_metrics,
        "attempts": attempts,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": Image.__version__,
            "numpy": np.__version__,
        },
    }
    result_path = args.output_dir / "result.json"
    serialized = json.dumps(result, indent=2, allow_nan=False)
    result_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
