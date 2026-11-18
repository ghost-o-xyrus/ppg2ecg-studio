import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import dsp  # noqa: E402
from bench.datasets import make_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def synth():
    """60 s synthetic PPG + ECG at 250 Hz (72 bpm)."""
    r = make_synthetic(duration=60.0, fs=250.0, hr=72.0, seed=1)
    return r


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


@pytest.fixture(scope="session")
def bidmc_dir():
    d = os.environ.get("BIDMC_DIR")
    if not d or not Path(d).exists():
        pytest.skip("set BIDMC_DIR to the folder holding bidmc_*_Signals.csv")
    return Path(d)


def write_csv(path, cols, header=None, delim=",", comments=()):
    with open(path, "w", encoding="utf-8") as f:
        for c in comments:
            f.write(c + "\n")
        if header:
            f.write(delim.join(header) + "\n")
        for row in zip(*cols):
            f.write(delim.join(f"{v:.6f}" for v in row) + "\n")
    return path


__all__ = ["write_csv", "dsp", "np"]
