import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from adversarial_image_attacks.cli import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: object) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    source_path = str(PROJECT_ROOT / "src")
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        source_path
        if not current_pythonpath
        else os.pathsep.join((source_path, current_pythonpath))
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "adversarial_image_attacks",
            *(str(arg) for arg in args),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_accepts_fractional_epsilon_and_all_merge_options(tmp_path: Path) -> None:
    base_path = tmp_path / "base.png"
    guide_path = tmp_path / "guide.png"
    output_path = tmp_path / "merged.png"
    Image.new("RGB", (2, 2), (100, 100, 100)).save(base_path)
    Image.new("RGB", (1, 3), (200, 200, 200)).save(guide_path)

    completed = run_cli(
        base_path,
        guide_path,
        output_path,
        "--strength",
        "1",
        "--epsilon",
        "1/255",
        "--resize-mode",
        "stretch",
        "--background",
        "1,2,3",
    )

    assert completed.returncode == 0, completed.stderr
    with Image.open(output_path) as merged:
        assert merged.size == (2, 2)
        assert list(merged.convert("RGB").getdata()) == [(101, 101, 101)] * 4


def test_cli_refuses_collision_until_force_is_given(tmp_path: Path) -> None:
    base_path = tmp_path / "base.png"
    guide_path = tmp_path / "guide.png"
    output_path = tmp_path / "merged.png"
    Image.new("RGB", (1, 1), (100, 100, 100)).save(base_path)
    Image.new("RGB", (1, 1), (200, 200, 200)).save(guide_path)

    first = run_cli(base_path, guide_path, output_path, "--epsilon", "1/255")
    assert first.returncode == 0, first.stderr
    first_bytes = output_path.read_bytes()

    Image.new("RGB", (1, 1), (0, 0, 0)).save(guide_path)
    refused = run_cli(base_path, guide_path, output_path, "--epsilon", "1/255")

    assert refused.returncode != 0
    assert output_path.read_bytes() == first_bytes

    forced = run_cli(
        base_path,
        guide_path,
        output_path,
        "--epsilon",
        "1/255",
        "--force",
    )
    assert forced.returncode == 0, forced.stderr
    with Image.open(output_path) as merged:
        assert merged.convert("RGB").getpixel((0, 0)) == (99, 99, 99)


def test_cli_rejects_non_png_output(tmp_path: Path) -> None:
    base_path = tmp_path / "base.png"
    guide_path = tmp_path / "guide.png"
    output_path = tmp_path / "merged.jpg"
    Image.new("RGB", (1, 1)).save(base_path)
    Image.new("RGB", (1, 1)).save(guide_path)

    completed = run_cli(base_path, guide_path, output_path)

    assert completed.returncode != 0
    assert not output_path.exists()


def test_cli_formats_decompression_bomb_errors_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base_path = tmp_path / "base.png"
    guide_path = tmp_path / "guide.png"
    output_path = tmp_path / "merged.png"
    Image.new("RGB", (2, 2)).save(base_path)
    Image.new("RGB", (2, 2)).save(guide_path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    with pytest.raises(SystemExit) as exit_info:
        main([str(base_path), str(guide_path), str(output_path)])

    captured = capsys.readouterr()
    assert exit_info.value.code == 2
    assert "error" in captured.err.lower()
    assert "Traceback" not in captured.err
