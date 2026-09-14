from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "examples"
    / "model-aware-pgd"
    / "run_attack.py"
)
SPEC = importlib.util.spec_from_file_location("run_attack", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUN_ATTACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_ATTACK)


def test_parse_levels_requires_unique_ascending_values() -> None:
    assert RUN_ATTACK._parse_levels("1,2,4,8") == [1, 2, 4, 8]
    with pytest.raises(argparse.ArgumentTypeError):
        RUN_ATTACK._parse_levels("2,1")


def test_metrics_reports_exact_uint8_budget() -> None:
    clean = np.zeros((8, 8, 3), dtype=np.uint8)
    candidate = clean.copy()
    candidate[0, 0] = [1, 2, 3]

    metrics = RUN_ATTACK._metrics(clean, candidate)

    assert metrics["linf_levels"] == 3
    assert metrics["linf_normalized"] == pytest.approx(3 / 255)
    assert metrics["changed_channels"] == 3
    assert metrics["total_channels"] == 192
