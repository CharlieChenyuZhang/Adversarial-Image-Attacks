"""Command-line interface for the bounded image merge."""

from __future__ import annotations

import argparse
import math
import sys
from typing import Sequence, Tuple

from PIL import Image

from .merge import merge_files


def _number_or_fraction(value: str) -> float:
    try:
        if "/" in value:
            numerator_text, denominator_text = value.split("/", maxsplit=1)
            number = float(numerator_text) / float(denominator_text)
        else:
            number = float(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(
            "expected a number or fraction such as 0.031 or 8/255"
        ) from exc
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("value must be finite")
    return number


def _rgb_color(value: str) -> Tuple[int, int, int]:
    try:
        parts = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected R,G,B integer values") from exc
    if len(parts) != 3 or any(channel < 0 or channel > 255 for channel in parts):
        raise argparse.ArgumentTypeError(
            "expected three comma-separated values between 0 and 255"
        )
    return parts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="advmerge",
        description=(
            "Move a base image toward a guide image while enforcing a per-channel "
            "L-infinity perturbation bound."
        ),
    )
    parser.add_argument("base", help="base image to preserve")
    parser.add_argument("guide", help="image that guides the perturbation")
    parser.add_argument("output", help="output path, which must end in .png")
    parser.add_argument(
        "--strength",
        type=_number_or_fraction,
        default=1.0,
        help="fraction to move toward the guide, from 0 to 1 (default: 1)",
    )
    parser.add_argument(
        "--epsilon",
        type=_number_or_fraction,
        default=8 / 255,
        help="normalized pixel bound, as a number or fraction (default: 8/255)",
    )
    parser.add_argument(
        "--resize-mode",
        choices=("cover", "stretch"),
        default="cover",
        help="how to resize a differently sized guide (default: cover)",
    )
    parser.add_argument(
        "--background",
        type=_rgb_color,
        default=(255, 255, 255),
        metavar="R,G,B",
        help="background for transparent pixels (default: 255,255,255)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        result = merge_files(
            arguments.base,
            arguments.guide,
            arguments.output,
            strength=arguments.strength,
            epsilon=arguments.epsilon,
            resize_mode=arguments.resize_mode,
            background=arguments.background,
            overwrite=arguments.force,
        )
    except (Image.DecompressionBombError, OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"advmerge: error: {exc}\n")

    print(
        f"Saved {result.output_path} "
        f"(max |delta|: {result.max_delta * 255:.0f}/255, "
        f"mean |delta|: {result.mean_delta * 255:.2f}/255)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
