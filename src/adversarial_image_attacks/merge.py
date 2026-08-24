"""Bounded merging of a base image and a guide image."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageOps

PathLike = Union[str, os.PathLike]
RGBColor = Tuple[int, int, int]
VALID_RESIZE_MODES = frozenset({"cover", "stretch"})
_WORKING_SET_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class MergeResult:
    """Summary of a file merge.

    ``max_delta`` and ``mean_delta`` are absolute RGB differences measured in
    normalized pixel space, where 1.0 is 255 pixel levels.
    """

    output_path: Path
    max_delta: float
    mean_delta: float


def _finite_float(value: float, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _validate_options(
    strength: float,
    epsilon: float,
    resize_mode: str,
    background: Sequence[int],
) -> Tuple[float, float, RGBColor]:
    strength_value = _finite_float(strength, "strength")
    if not 0.0 <= strength_value <= 1.0:
        raise ValueError("strength must be between 0 and 1")

    epsilon_value = _finite_float(epsilon, "epsilon")
    if not 0.0 <= epsilon_value <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")

    if resize_mode not in VALID_RESIZE_MODES:
        choices = ", ".join(sorted(VALID_RESIZE_MODES))
        raise ValueError(f"resize_mode must be one of: {choices}")

    if isinstance(background, (str, bytes)) or len(background) != 3:
        raise ValueError("background must contain exactly three RGB values")

    channels = []
    for value in background:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("background RGB values must be integers")
        channel = int(value)
        if not 0 <= channel <= 255:
            raise ValueError("background RGB values must be between 0 and 255")
        channels.append(channel)

    return strength_value, epsilon_value, (channels[0], channels[1], channels[2])


def _to_rgb(image: Image.Image, background: RGBColor) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("base and guide must be PIL Image objects")

    transposed = ImageOps.exif_transpose(image)
    if transposed.mode == "La":
        # Pillow cannot convert premultiplied grayscale-alpha directly to RGBA.
        transposed = transposed.convert("LA")
    has_transparency = (
        "A" in transposed.getbands()
        or "a" in transposed.getbands()
        or "transparency" in transposed.info
    )
    if not has_transparency:
        return transposed.convert("RGB")

    foreground = transposed.convert("RGBA")
    backdrop = Image.new("RGBA", foreground.size, (*background, 255))
    return Image.alpha_composite(backdrop, foreground).convert("RGB")


def _resize_guide(
    guide: Image.Image, size: Tuple[int, int], resize_mode: str
) -> Image.Image:
    if guide.size == size:
        return guide
    if resize_mode == "cover":
        return ImageOps.fit(
            guide,
            size,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
    return guide.resize(size, resample=Image.Resampling.LANCZOS)


def _rows_per_chunk(width: int, bytes_per_channel: int) -> int:
    bytes_per_row = width * 3 * bytes_per_channel
    return max(1, _WORKING_SET_BYTES // max(1, bytes_per_row))


def _bounded_merge_pixels(
    base_pixels: np.ndarray,
    guide_pixels: np.ndarray,
    strength: float,
    epsilon: float,
) -> np.ndarray:
    """Merge uint8 RGB arrays without allocating full-size float64 copies."""

    output = np.empty_like(base_pixels)
    epsilon_pixels = epsilon * 255.0
    # Integer output values can move only this many complete pixel levels while
    # remaining inside the normalized epsilon interval.
    integer_epsilon = math.floor(epsilon_pixels)
    rows_per_chunk = _rows_per_chunk(
        base_pixels.shape[1], np.dtype(np.float32).itemsize
    )

    for first_row in range(0, base_pixels.shape[0], rows_per_chunk):
        last_row = min(first_row + rows_per_chunk, base_pixels.shape[0])
        base_chunk = base_pixels[first_row:last_row]

        candidate = guide_pixels[first_row:last_row].astype(np.float32)
        np.subtract(candidate, base_chunk, out=candidate)
        np.multiply(candidate, strength, out=candidate)
        np.clip(candidate, -epsilon_pixels, epsilon_pixels, out=candidate)
        np.add(candidate, base_chunk, out=candidate)
        np.rint(candidate, out=candidate)

        # Clamp once more in integer space so PNG round trips retain the bound
        # even when epsilon is not an exact multiple of 1/255.
        np.subtract(candidate, base_chunk, out=candidate)
        np.clip(candidate, -integer_epsilon, integer_epsilon, out=candidate)
        integer_chunk = candidate.astype(np.int16)
        np.add(integer_chunk, base_chunk, out=integer_chunk)
        np.clip(integer_chunk, 0, 255, out=integer_chunk)
        output[first_row:last_row] = integer_chunk

    return output


def _delta_statistics(
    base_pixels: np.ndarray, merged_pixels: np.ndarray
) -> Tuple[float, float]:
    maximum = 0
    total = 0
    count = int(base_pixels.size)
    rows_per_chunk = _rows_per_chunk(base_pixels.shape[1], np.dtype(np.int16).itemsize)

    for first_row in range(0, base_pixels.shape[0], rows_per_chunk):
        last_row = min(first_row + rows_per_chunk, base_pixels.shape[0])
        difference = merged_pixels[first_row:last_row].astype(np.int16)
        np.subtract(difference, base_pixels[first_row:last_row], out=difference)
        np.abs(difference, out=difference)
        maximum = max(maximum, int(difference.max(initial=0)))
        total += int(difference.sum(dtype=np.int64))

    return maximum / 255.0, (total / count / 255.0) if count else 0.0


def merge_images(
    base: Image.Image,
    guide: Image.Image,
    *,
    strength: float = 1.0,
    epsilon: float = 8 / 255,
    resize_mode: str = "cover",
    background: Sequence[int] = (255, 255, 255),
) -> Image.Image:
    """Move ``base`` toward ``guide`` within an L-infinity pixel bound.

    The guide is resized to the EXIF-corrected base dimensions. Both images are
    converted to RGB, with transparent pixels composited over ``background``.
    The returned image is an in-memory, eight-bit RGB image.

    ``epsilon`` uses normalized pixel units. For example, ``8 / 255`` limits
    every saved RGB channel to at most eight pixel levels from the base.
    """

    strength_value, epsilon_value, background_rgb = _validate_options(
        strength, epsilon, resize_mode, background
    )
    base_rgb = _to_rgb(base, background_rgb)
    guide_rgb = _resize_guide(
        _to_rgb(guide, background_rgb), base_rgb.size, resize_mode
    )

    base_pixels = np.asarray(base_rgb, dtype=np.uint8)
    guide_pixels = np.asarray(guide_rgb, dtype=np.uint8)
    quantized = _bounded_merge_pixels(
        base_pixels, guide_pixels, strength_value, epsilon_value
    )

    return Image.fromarray(quantized)


def merge_files(
    base_path: PathLike,
    guide_path: PathLike,
    output_path: PathLike,
    *,
    strength: float = 1.0,
    epsilon: float = 8 / 255,
    resize_mode: str = "cover",
    background: Sequence[int] = (255, 255, 255),
    overwrite: bool = False,
) -> MergeResult:
    """Merge two image files and atomically save a lossless PNG."""

    base_file = Path(base_path).expanduser()
    guide_file = Path(guide_path).expanduser()
    output_file = Path(output_path).expanduser()

    if output_file.suffix.lower() != ".png":
        raise ValueError("output must use the .png extension")

    output_absolute = output_file.resolve(strict=False)
    input_absolutes = {
        base_file.resolve(strict=False),
        guide_file.resolve(strict=False),
    }
    if output_absolute in input_absolutes:
        raise ValueError("output path must be different from both input paths")
    if os.path.lexists(output_file) and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_file} (pass overwrite=True to replace it)"
        )

    with Image.open(base_file) as base_image, Image.open(guide_file) as guide_image:
        merged = merge_images(
            base_image,
            guide_image,
            strength=strength,
            epsilon=epsilon,
            resize_mode=resize_mode,
            background=background,
        )
        _, _, background_rgb = _validate_options(
            strength, epsilon, resize_mode, background
        )
        base_rgb = _to_rgb(base_image, background_rgb)
        base_pixels = np.asarray(base_rgb, dtype=np.uint8)

    merged_pixels = np.asarray(merged, dtype=np.uint8)
    max_delta, mean_delta = _delta_statistics(base_pixels, merged_pixels)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_file.stem}.",
            suffix=".png",
            dir=output_file.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            merged.save(temporary, format="PNG", optimize=False)
        if overwrite:
            os.replace(temporary_path, output_file)
        else:
            try:
                # A hard-link publish is atomic and fails if any directory entry,
                # including a dangling symlink, appeared after the earlier check.
                os.link(temporary_path, output_file)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"output already exists: {output_file} "
                    "(pass overwrite=True to replace it)"
                ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return MergeResult(
        output_path=output_file,
        max_delta=max_delta,
        mean_delta=mean_delta,
    )
