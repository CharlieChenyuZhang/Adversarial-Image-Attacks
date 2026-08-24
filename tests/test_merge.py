import os
from pathlib import Path

import pytest
from PIL import Image

import adversarial_image_attacks.merge as merge_module
from adversarial_image_attacks.merge import merge_files, merge_images


def test_merge_images_moves_toward_guide_within_epsilon() -> None:
    base = Image.new("RGB", (2, 2), (100, 100, 100))
    guide = Image.new("RGB", (2, 2), (200, 50, 100))

    merged = merge_images(base, guide, strength=1.0, epsilon=2 / 255)

    assert merged.mode == "RGB"
    assert merged.size == base.size
    assert list(merged.getdata()) == [(102, 98, 100)] * 4
    assert list(base.getdata()) == [(100, 100, 100)] * 4


def test_zero_strength_returns_an_independent_copy_of_base() -> None:
    base = Image.new("RGB", (2, 1))
    base.putdata([(10, 20, 30), (40, 50, 60)])
    guide = Image.new("RGB", (1, 3), (255, 255, 255))

    merged = merge_images(base, guide, strength=0.0, epsilon=1.0)

    assert merged is not base
    assert merged.mode == "RGB"
    assert merged.size == base.size
    assert list(merged.getdata()) == list(base.getdata())


def test_strength_controls_unsaturated_blend() -> None:
    base = Image.new("RGB", (1, 1), (20, 40, 60))
    guide = Image.new("RGB", (1, 1), (100, 120, 140))

    merged = merge_images(base, guide, strength=0.25, epsilon=1.0)

    assert merged.getpixel((0, 0)) == (40, 60, 80)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"strength": -0.01}, "strength"),
        ({"strength": 1.01}, "strength"),
        ({"epsilon": -0.01}, "epsilon"),
        ({"epsilon": 1.01}, "epsilon"),
        ({"epsilon": float("nan")}, "finite"),
        ({"background": (0, 0, 256)}, "background"),
    ],
)
def test_merge_images_rejects_invalid_options(options: object, message: str) -> None:
    image = Image.new("RGB", (1, 1))

    with pytest.raises((TypeError, ValueError), match=message):
        merge_images(image, image, **options)  # type: ignore[arg-type]


def test_background_is_used_when_flattening_transparency() -> None:
    base = Image.new("RGBA", (1, 1), (0, 0, 0, 255))
    guide = Image.new("RGBA", (1, 1), (255, 0, 0, 0))

    merged = merge_images(
        base,
        guide,
        strength=1.0,
        epsilon=1.0,
        background=(12, 34, 56),
    )

    assert merged.getpixel((0, 0)) == (12, 34, 56)


def test_partial_and_premultiplied_alpha_are_composited() -> None:
    base = Image.new("RGB", (1, 1), (0, 0, 0))
    rgba = Image.new("RGBA", (1, 1), (200, 100, 0, 128))
    premultiplied_rgba = Image.new("RGBa", (1, 1), (100, 50, 0, 128))
    premultiplied_grayscale = Image.new("La", (1, 1), (64, 128))
    palette = Image.new("P", (1, 1), 0)
    palette.putpalette([255, 0, 0] + [0, 0, 0] * 255)
    palette.info["transparency"] = bytes([128] + [255] * 255)

    assert merge_images(
        base, rgba, epsilon=1.0, background=(0, 0, 0)
    ).getpixel((0, 0)) == (100, 50, 0)
    assert merge_images(
        base, premultiplied_rgba, epsilon=1.0, background=(0, 0, 0)
    ).getpixel((0, 0)) == (100, 50, 0)
    assert merge_images(
        base, premultiplied_grayscale, epsilon=1.0, background=(0, 0, 0)
    ).getpixel((0, 0)) == (64, 64, 64)
    assert merge_images(
        base, palette, epsilon=1.0, background=(0, 0, 0)
    ).getpixel((0, 0)) == (128, 0, 0)


def test_cover_and_stretch_resize_to_base_but_use_different_geometry() -> None:
    base = Image.new("RGB", (2, 4), (0, 0, 0))
    guide = Image.new("RGB", (4, 2))
    guide.putdata(
        [
            (255, 0, 0),
            (0, 255, 0),
            (0, 255, 0),
            (0, 0, 255),
        ]
        * 2
    )

    covered = merge_images(
        base, guide, strength=1.0, epsilon=1.0, resize_mode="cover"
    )
    stretched = merge_images(
        base, guide, strength=1.0, epsilon=1.0, resize_mode="stretch"
    )

    assert covered.size == base.size
    assert stretched.size == base.size
    assert covered.tobytes() != stretched.tobytes()


def test_merge_images_rejects_unknown_resize_mode() -> None:
    image = Image.new("RGB", (1, 1))

    with pytest.raises(ValueError, match="resize"):
        merge_images(image, image, resize_mode="contain")  # type: ignore[arg-type]


def test_merge_files_writes_png_and_reports_normalized_deltas(tmp_path: Path) -> None:
    base_path = tmp_path / "base.png"
    guide_path = tmp_path / "guide.png"
    output_path = tmp_path / "merged.png"
    Image.new("RGB", (2, 2), (100, 100, 100)).save(base_path)
    Image.new("RGB", (2, 2), (200, 200, 200)).save(guide_path)

    result = merge_files(
        base_path,
        guide_path,
        output_path,
        strength=1.0,
        epsilon=2 / 255,
    )

    assert Path(result.output_path) == output_path
    assert result.max_delta == pytest.approx(2 / 255)
    assert result.mean_delta == pytest.approx(2 / 255)
    with Image.open(output_path) as merged:
        assert merged.format == "PNG"
        assert merged.mode == "RGB"
        assert list(merged.getdata()) == [(102, 102, 102)] * 4


def test_fractional_epsilon_remains_strict_after_png_round_trip(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.png"
    guide_path = tmp_path / "guide.png"
    output_path = tmp_path / "merged.png"
    Image.new("RGB", (1, 1), (100, 100, 100)).save(base_path)
    Image.new("RGB", (1, 1), (255, 0, 255)).save(guide_path)

    result = merge_files(base_path, guide_path, output_path, epsilon=7.5 / 255)

    with Image.open(output_path) as merged:
        assert merged.getpixel((0, 0)) == (107, 93, 107)
    assert result.max_delta == pytest.approx(7 / 255)


def test_epsilon_just_below_an_integer_level_is_not_rounded_up() -> None:
    base = Image.new("RGB", (1, 1), (100, 100, 100))
    guide = Image.new("RGB", (1, 1), (255, 255, 255))

    merged = merge_images(base, guide, epsilon=(8 / 255) - 1e-14)

    assert merged.getpixel((0, 0)) == (107, 107, 107)


def test_merge_files_never_replaces_an_input(tmp_path: Path) -> None:
    base_path = tmp_path / "base.png"
    guide_path = tmp_path / "guide.png"
    Image.new("RGB", (1, 1), (100, 100, 100)).save(base_path)
    Image.new("RGB", (1, 1), (200, 200, 200)).save(guide_path)
    original = base_path.read_bytes()

    with pytest.raises(ValueError, match="different"):
        merge_files(base_path, guide_path, base_path, overwrite=True)

    assert base_path.read_bytes() == original


def test_merge_files_applies_base_exif_orientation(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jpg"
    guide_path = tmp_path / "guide.png"
    output_path = tmp_path / "merged.png"
    base = Image.new("RGB", (2, 1), (100, 100, 100))
    orientation = Image.Exif()
    orientation[274] = 6
    base.save(base_path, exif=orientation)
    Image.new("RGB", (1, 2), (200, 200, 200)).save(guide_path)

    merge_files(base_path, guide_path, output_path, epsilon=0)

    with Image.open(output_path) as merged:
        assert merged.size == (1, 2)


def test_no_overwrite_publish_is_atomic_against_a_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_path = tmp_path / "base.png"
    guide_path = tmp_path / "guide.png"
    output_path = tmp_path / "merged.png"
    Image.new("RGB", (1, 1), (100, 100, 100)).save(base_path)
    Image.new("RGB", (1, 1), (200, 200, 200)).save(guide_path)
    real_link = os.link

    def competing_publish(source: object, destination: object) -> None:
        Path(destination).write_bytes(b"written by another process")
        real_link(source, destination)

    monkeypatch.setattr(merge_module.os, "link", competing_publish)

    with pytest.raises(FileExistsError):
        merge_files(base_path, guide_path, output_path)

    assert output_path.read_bytes() == b"written by another process"


def test_dangling_output_symlink_is_not_replaced(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symbolic links are unavailable")
    base_path = tmp_path / "base.png"
    guide_path = tmp_path / "guide.png"
    output_path = tmp_path / "merged.png"
    missing_target = tmp_path / "missing.png"
    Image.new("RGB", (1, 1)).save(base_path)
    Image.new("RGB", (1, 1), (255, 255, 255)).save(guide_path)
    output_path.symlink_to(missing_target)

    with pytest.raises(FileExistsError):
        merge_files(base_path, guide_path, output_path)

    assert output_path.is_symlink()
    assert output_path.readlink() == missing_target


def test_merge_files_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    base_path = tmp_path / "base.png"
    guide_path = tmp_path / "guide.png"
    output_path = tmp_path / "merged.png"
    Image.new("RGB", (1, 1), (0, 0, 0)).save(base_path)
    Image.new("RGB", (1, 1), (255, 255, 255)).save(guide_path)
    output_path.write_bytes(b"keep me")

    with pytest.raises(FileExistsError):
        merge_files(base_path, guide_path, output_path)

    assert output_path.read_bytes() == b"keep me"


def test_merge_files_requires_a_png_output_path(tmp_path: Path) -> None:
    base_path = tmp_path / "base.png"
    guide_path = tmp_path / "guide.png"
    Image.new("RGB", (1, 1)).save(base_path)
    Image.new("RGB", (1, 1)).save(guide_path)

    with pytest.raises(ValueError, match="[Pp][Nn][Gg]"):
        merge_files(base_path, guide_path, tmp_path / "merged.jpg")
