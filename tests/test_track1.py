"""The app must score exactly as Track 1: BIDMC subjects 01, 05, 27, whole 8-min recordings, through the app's
own intake (CSV reader, rate from the time column, 128 Hz conversion) and engine.

    set BIDMC_DIR=...\\bidmc_csv          (folder with bidmc_*_Signals.csv)
    python -m pytest tests/test_track1.py

Zhu DCT is checked only when its CapnoBase coefficients are installed (fit_on == capnobase).
Reference values: results/track1/subjects_cons.csv of the thesis repository (clean condition). The thesis
stored predictions in float16 before scoring, so values agree to about 1e-3, not exactly.
"""
import numpy as np
import pytest

from studio import config, engine, intake

TRACK1 = {  # model -> subject -> (hr_ae, r_f1_dc, beat_pearson, pearson_lag, fd, rmssd_ae)
    "template": {"01": (1.207335, 0.987603, 0.919675, 0.785963, 1.642969, 56.492493),
                 "05": (2.201089, 0.978121, 0.767107, 0.548994, 1.668548, 76.842060),
                 "27": (1.438040, 0.983630, 0.714764, 0.684773, 1.752538, 10.224675)},
    "zhu_dct": {"01": (0.943613, 0.982094, 0.815268, 0.626144, 1.598886, 67.535702),
                "05": (2.635263, 0.978065, 0.760663, 0.651900, 1.745727, 50.290573),
                "27": (1.616345, 0.982181, 0.641902, 0.609145, 1.616519, 18.622789)},
}
KEYS = ("hr_ae", "r_f1_dc", "beat_pearson", "pearson_lag", "fd", "rmssd_ae")
TOL = {"hr_ae": 0.02, "r_f1_dc": 0.003, "beat_pearson": 0.003, "pearson_lag": 0.003, "fd": 0.01, "rmssd_ae": 0.5}


def _models():
    out = []
    for s in engine.discover():
        if s.id not in TRACK1:
            continue
        lm = engine.load_model(s)
        if lm.status != engine.READY:
            continue
        if s.id == "zhu_dct" and lm.info.get("fit_on") != "capnobase":
            continue
        out.append(lm)
    return out


@pytest.mark.parametrize("subject", ["01", "05", "27"])
def test_matches_track1(bidmc_dir, subject):
    f = bidmc_dir / f"bidmc_{subject}_Signals.csv"
    p = intake.prepare(f, "ppg", limit_s=None, limit_mb=None)
    assert p.fs_in == pytest.approx(125.0) and p.column.strip() == "PLETH" and not p.flipped
    e = intake.prepare_embedded(p, next(iter(p.embedded)), limit_s=None)
    assert e.column.strip() == "II"
    ppg, ecg, _ = intake.align(p, e)
    models = _models()
    assert models, "template baseline must always load"
    for lm in models:
        r = engine.run_model(lm, ppg, intake.aligned_peaks(p, ppg), ecg)
        for k, ref in zip(KEYS, TRACK1[lm.spec.id][subject]):
            got = r.vs_true[k]
            assert abs(got - ref) <= TOL[k], f"{lm.spec.id} {subject} {k}: app {got:.6f} vs Track 1 {ref:.6f}"


def test_window_settings_are_track1():
    assert config.EVAL_WINDOW_S == 10.0 and config.FD_MAX_WINDOWS == 20 and config.STITCH_OVERLAP == 0.5
