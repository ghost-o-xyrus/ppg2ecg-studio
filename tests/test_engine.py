"""Engine end to end on synthetic data: registry, both built-in models, scoring, PPG-only measures."""
from pathlib import Path

import numpy as np
import pytest

from bench import dsp
from bench.datasets import make_synthetic
from studio import config, engine, scoring


@pytest.fixture(scope="module")
def zhu_weights(tmp_path_factory):
    """Zhu DCT fitted on synthetic records, so the test needs no dataset."""
    from bench.models.zhu_dct import ZhuDCT
    d = tmp_path_factory.mktemp("weights")
    m = ZhuDCT()
    m.fit([make_synthetic(subject=str(i), duration=120, fs=256, hr=60 + 8 * i, seed=i) for i in range(4)])
    (d / "zhu_dct").mkdir()
    np.savez(d / "zhu_dct" / "zhu_capnobase.npz", W=m.W, b=m.b, pat=m.pat, n_cycles=1, fit_on="synthetic")
    return d


def test_discover_lists_builtins():
    ids = [s.id for s in engine.discover()]
    assert ids[:2] == ["template", "zhu_dct"] and "my_model" not in ids
    t = engine.discover()[0]
    assert t.always_installed and t.installed


def test_missing_weights_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path)
    z = [s for s in engine.discover() if s.id == "zhu_dct"][0]
    lm = engine.load_model(z)
    assert lm.status == engine.MISSING and "zhu_capnobase.npz" in lm.error


def test_run_both_models(monkeypatch, zhu_weights):
    monkeypatch.setattr(config, "WEIGHTS_DIR", zhu_weights)
    r = make_synthetic(subject="t", duration=40, fs=250, hr=75, seed=9)
    ppg = dsp.resample(r.ppg, r.ppg_fs, 128)
    ecg = dsp.resample(r.ecg, r.ecg_fs, 128)
    peaks = dsp.detect_ppg_peaks(ppg, 128)
    for s in [s for s in engine.discover() if s.id in ("template", "zhu_dct")]:   # the two NumPy baselines
        lm = engine.load_model(s)
        assert lm.status == engine.READY, lm.error
        res = engine.run_model(lm, ppg, peaks, ecg)
        assert len(res.pred) == len(ppg) and np.isfinite(res.pred).all()
        assert res.gen_s > 0 and res.rt_factor < 1
        assert res.vs_ppg["hr_recon"] == pytest.approx(75, abs=3)
        assert res.vs_ppg["pulse_f1"] > 0.9
        assert res.vs_true["r_f1_dc"] > 0.9 and res.vs_true["hr_ae"] < 3
        assert len(res.row(scoring.ERRORS)) == len(scoring.ERRORS)


def test_ppg_only_measures_and_flags():
    pk = np.arange(100, 128 * 30, 100)
    flat = scoring.score_vs_ppg(np.zeros(128 * 30), pk)
    assert "flat" in flat["flags"] and flat["pulse_f1"] == 0.0
    assert scoring.best_index([1.0, 0.5, np.nan], "low") == 1
    assert scoring.best_index([1.0, 0.5, np.nan], "high") == 0
    assert scoring.best_index([-0.2, 0.1], "zero") == 1
    assert scoring.fmt(float("nan"), scoring.ERRORS[0]) == "-"


def test_display_shift():
    assert scoring.display_shift({"r_offset_ms": 250.0}) == 32
    assert scoring.display_shift({}) == 0


def test_models_load_in_parallel():
    """The window loads every model on its own thread at once; their imports must not deadlock."""
    import subprocess
    import sys
    code = ("import sys; sys.path.insert(0, '.'); from concurrent.futures import ThreadPoolExecutor\n"
            "from studio import engine\n"
            "specs = [s for s in engine.discover() if s.installed] * 4\n"
            "with ThreadPoolExecutor(len(specs)) as ex:\n"
            "    lms = list(ex.map(engine.load_model, specs))\n"
            "bad = [(m.spec.id, m.error) for m in lms if m.status != engine.READY]\n"
            "print(bad); sys.exit(1 if bad else 0)\n")
    for _ in range(3):        # a fresh interpreter each time: nothing imported yet, as when the app starts
        r = subprocess.run([sys.executable, "-c", code], cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr
