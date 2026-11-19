"""The app must score exactly as Track 1: BIDMC subjects 01, 05, 27, whole 8-min recordings, through the app's
own intake (CSV reader, rate from the time column, 128 Hz conversion, the file-rate signal for the models) and
engine, for every installed method.

    set BIDMC_DIR=...\bidmc_csv          (folder with bidmc_*_Signals.csv)
    python -m pytest tests/test_track1.py

Models that are not installed are skipped; Zhu DCT is checked only when its CapnoBase coefficients are
installed (fit_on == capnobase). Reference values: results/track1/subjects_cons.csv of the thesis repository
(clean condition). The thesis stored predictions in float16 before scoring, so values agree to about 1e-3,
not exactly.

PPGFlowECG is stochastic and its Track 1 run drew its noise on a CUDA GPU, which gives different random
numbers than the CPU for the same seed, so the app's seed-0 sample is a different sample of the same model.
Its tolerance is 3.5 standard deviations of the difference between two independent samples (3.5 x sqrt(2) x
the seed-to-seed SD, measured over seeds 0-8 on BIDMC 01/05/27 with this app: hr_ae 0.42, r_f1_dc 0.0035,
beat_pearson 0.027, pearson_lag 0.020, fd 0.027, rmssd_ae 16.9). RDDM is stochastic too, but Track 1 sampled
it on the CPU with the same seed, so it is held to the exact tolerances.
"""
import pytest

from studio import config, engine, intake

TRACK1 = {  # model -> subject -> (hr_ae, r_f1_dc, beat_pearson, pearson_lag, fd, rmssd_ae)
    "template": {"01": (1.207335, 0.987603, 0.919675, 0.785963, 1.642969, 56.492493),
                 "05": (2.201089, 0.978121, 0.767107, 0.548994, 1.668548, 76.842060),
                 "27": (1.438040, 0.983630, 0.714764, 0.684773, 1.752538, 10.224675)},
    "zhu_dct": {"01": (0.943613, 0.982094, 0.815268, 0.626144, 1.598886, 67.535702),
                "05": (2.635263, 0.978065, 0.760663, 0.651900, 1.745727, 50.290573),
                "27": (1.616345, 0.982181, 0.641902, 0.609145, 1.616519, 18.622789)},
    "chiu2020": {"01": (3.292614, 0.977654, 0.851817, 0.781741, 1.634875, 53.231538),
                 "05": (3.269015, 0.976593, 0.891234, 0.781270, 1.567843, 117.002489),
                 "27": (1.868765, 0.979943, 0.732870, 0.707703, 0.780673, 25.211397)},
    "clep_gan": {"01": (0.776216, 0.991065, 0.964541, 0.851671, 1.929991, 30.799164),
                 "05": (2.537273, 0.980013, 0.958127, 0.830534, 1.827839, 63.058313),
                 "27": (1.136720, 0.983676, 0.944342, 0.874243, 1.137843, 17.264244)},
    "p2e_wgan": {"01": (15.045571, 0.867614, 0.297219, 0.285489, 1.748612, 400.394825),
                 "05": (14.162476, 0.893646, 0.222943, 0.308014, 1.772020, 342.553483),
                 "27": (0.791550, 0.976173, 0.257217, 0.426993, 1.910937, 56.221701)},
    "yalcin2026": {"01": (24.225722, 0.405439, 0.339290, 0.199447, 1.816213, 335.694247),
                   "05": (20.304172, 0.718713, 0.323513, 0.228963, 1.811039, 410.226517),
                   "27": (22.001459, 0.444152, 0.069327, 0.218648, 1.822908, 373.897901)},
    "cardiogan": {"01": (1.712954, 0.982044, 0.723372, 0.157899, 1.881091, 90.827402),
                  "05": (4.519429, 0.963542, 0.752411, 0.159059, 1.817792, 158.570126),
                  "27": (1.456386, 0.985075, 0.513479, 0.136732, 1.502191, 31.309960)},
    "rddm": {"01": (1.394836, 0.982019, 0.672258, 0.440524, 1.797468, 42.595587),
             "05": (4.664410, 0.949020, 0.633551, 0.251035, 1.759841, 168.049750),
             "27": (7.138694, 0.792424, 0.496999, 0.206432, 1.475088, 151.693992)},
    "ppgflowecg": {"01": (0.430396, 0.978487, 0.662813, 0.572467, 1.878714, 35.294810),
                   "05": (0.238067, 0.979194, 0.683560, 0.596732, 1.773793, 6.855179),
                   "27": (0.566363, 0.984968, 0.456652, 0.514382, 1.641015, 28.149054)},
}
KEYS = ("hr_ae", "r_f1_dc", "beat_pearson", "pearson_lag", "fd", "rmssd_ae")
TOL = {"hr_ae": 0.02, "r_f1_dc": 0.003, "beat_pearson": 0.003, "pearson_lag": 0.003, "fd": 0.01, "rmssd_ae": 0.5}
TOL_SAMPLED = {"ppgflowecg": {"hr_ae": 2.1, "r_f1_dc": 0.018, "beat_pearson": 0.13, "pearson_lag": 0.10, "fd": 0.14,
                              "rmssd_ae": 84.0}}


@pytest.fixture(scope="module")
def models():
    out = []
    for s in engine.discover():
        if s.id not in TRACK1 or not s.installed:
            continue
        lm = engine.load_model(s)
        assert lm.status == engine.READY, f"{s.id}: {lm.error}"
        if s.id == "zhu_dct" and lm.info.get("fit_on") != "capnobase":
            continue
        out.append(lm)
    assert out, "template baseline must always load"
    return out


@pytest.mark.parametrize("subject", ["01", "05", "27"])
def test_matches_track1(bidmc_dir, models, subject):
    f = bidmc_dir / f"bidmc_{subject}_Signals.csv"
    p = intake.prepare(f, "ppg", limit_s=None, limit_mb=None)
    assert p.fs_in == pytest.approx(125.0) and p.column.strip() == "PLETH" and not p.flipped
    e = intake.prepare_embedded(p, next(iter(p.embedded)), limit_s=None)
    assert e.column.strip() == "II"
    ppg, ecg, _ = intake.align(p, e)
    src = intake.model_input(p, e, ppg)
    bad = []
    for lm in models:
        r = engine.run_model(lm, ppg, intake.aligned_peaks(p, ppg), ecg, seed=0, src=src)
        tol = TOL_SAMPLED.get(lm.spec.id, TOL)
        for k, ref in zip(KEYS, TRACK1[lm.spec.id][subject]):
            got = r.vs_true[k]
            if not abs(got - ref) <= tol[k]:
                bad.append(f"{lm.spec.id} {subject} {k}: app {got:.6f} vs Track 1 {ref:.6f}")
    assert not bad, "; ".join(bad)


def test_window_settings_are_track1():
    assert config.EVAL_WINDOW_S == 10.0 and config.FD_MAX_WINDOWS == 20 and config.STITCH_OVERLAP == 0.5
