"""Dataset readers used by the tests and tools/fit_baselines.py (trimmed copy of the thesis bench: BIDMC,
CapnoBase, synthetic). The app itself reads files through studio.intake.

Dataset adapters -> canonical Records.

Each loader is a generator `load_<name>(root, **kw) -> Iterator[Record]` that reads
the dataset in the layout it is distributed in (no manual conversion needed).
Register new loaders in LOADERS.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from ..record import Record, finite_segments


def _find_col(df, *needles):
    for c in df.columns:
        lc = c.strip().lower()
        if any(n in lc for n in needles):
            return c
    raise KeyError(f"none of {needles} in columns {list(df.columns)}")


# ------------------------------------------------------------------ BIDMC
def load_bidmc(root, **_) -> Iterator[Record]:
    """PhysioNet BIDMC, CSV flavour: <root>/bidmc_csv/bidmc_XX_Signals.csv (125 Hz)."""
    root = Path(root)
    files = sorted(root.rglob("bidmc_*_Signals.csv"))
    if not files:
        raise FileNotFoundError(f"no bidmc_*_Signals.csv under {root}")
    for f in files:
        df = pd.read_csv(f)
        sid = f.name.split("_")[1]
        yield Record("bidmc", sid, df[_find_col(df, "pleth")].to_numpy(), 125.0,
                     df[_find_col(df, " ii", "ii")].to_numpy(), 125.0, ppg_site="finger", ecg_lead="II")


# ------------------------------------------------------------------ CapnoBase
CAPNO_FS = 300.0
CAPNO_SECONDS = 480.0


def _capno_from_table(f):
    """*_8min_signal.csv (original) or *_8min_signal.tab (Borealis/Dataverse tab-separated copy)."""
    df = pd.read_csv(f, sep="\t" if f.suffix == ".tab" else ",")
    if df.shape[1] == 1:  # wrong delimiter guess -> let pandas sniff it
        df = pd.read_csv(f, sep=None, engine="python")
    return df[_find_col(df, "pleth")].to_numpy(float), df[_find_col(df, "ecg")].to_numpy(float)


def _mat_get(node, *path):
    for p in path:
        node = node[p] if isinstance(node, dict) or hasattr(node, "keys") else getattr(node, p)
    return node


def _capno_from_mat(f):
    """*_8min.mat: struct signal.pleth.y / signal.ecg.y, param.samplingrate.{pleth,ecg}.
    MATLAB v5/v7 via scipy; v7.3 (HDF5) via h5py."""
    try:
        from scipy.io import loadmat
        m = loadmat(f, squeeze_me=True, struct_as_record=False)
        get = lambda *p: np.asarray(_mat_get(m, *p), dtype=float).ravel()  # noqa: E731
    except NotImplementedError:  # v7.3
        import h5py
        h = h5py.File(f, "r")
        get = lambda *p: np.asarray(_mat_get(h, *p)[()], dtype=float).ravel()  # noqa: E731
    ppg, ecg = get("signal", "pleth", "y"), get("signal", "ecg", "y")
    try:
        fs_p, fs_e = float(get("param", "samplingrate", "pleth")[0]), float(get("param", "samplingrate", "ecg")[0])
    except Exception:
        fs_p = fs_e = CAPNO_FS
    return ppg, fs_p, ecg, fs_e


def load_capnobase(root, **_) -> Iterator[Record]:
    """CapnoBase IEEE TBME RR benchmark (42 x 8 min, 300 Hz), in whichever form you downloaded it:
    XXXX_8min_signal.csv (original), XXXX_8min_signal.tab (Borealis copy) or XXXX_8min.mat.
    One file per subject is used, in that order of preference."""
    root = Path(root)
    by_subject = {}
    for pattern in ("*_8min_signal.csv", "*_8min_signal.tab", "*_8min.mat"):
        for f in sorted(root.rglob(pattern)):
            by_subject.setdefault(f.name.split("_")[0], f)
    if not by_subject:
        raise FileNotFoundError(f"no *_8min_signal.csv / *_8min_signal.tab / *_8min.mat under {root}")
    for sid, f in sorted(by_subject.items()):
        if f.suffix == ".mat":
            ppg, fs_p, ecg, fs_e = _capno_from_mat(f)
        else:
            ppg, ecg = _capno_from_table(f)
            fs_p = fs_e = CAPNO_FS
        for name, x, fs in (("PPG", ppg, fs_p), ("ECG", ecg, fs_e)):  # every recording is 8 min
            if abs(len(x) / fs - CAPNO_SECONDS) > 5:
                print(f"[capnobase] warning {f.name}: {name} lasts {len(x) / fs:.1f}s at {fs} Hz (expected 480 s)")
        yield Record("capnobase", sid, ppg, fs_p, ecg, fs_e, ppg_site="finger", ecg_lead="unspecified",
                     meta={"source_file": f.name})



# ------------------------------------------------------------------ synthetic (tests / sanity)
def synth_beat(t, rr):
    """McSharry-like PQRST built from Gaussians; t in seconds relative to R peak."""
    s = np.sqrt(rr / 0.8)  # QT scales with sqrt(RR) (Bazett)
    comps = [(-0.20 * s, 0.025, 0.12), (-0.035, 0.010, -0.15), (0.0, 0.011, 1.0),
             (0.035, 0.010, -0.25), (0.26 * s, 0.045 * s, 0.30)]
    return sum(a * np.exp(-0.5 * ((t - mu) / sd) ** 2) for mu, sd, a in comps)


def make_synthetic(subject="00", duration=120.0, fs=256.0, hr=72.0, hrv=0.04, pat=0.25,
                   noise=0.01, seed=0) -> Record:
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    beats, tt = [], 0.5
    while tt < duration:
        beats.append(tt)
        tt += 60.0 / hr * (1 + hrv * rng.standard_normal())
    beats = np.array(beats)
    ecg = np.zeros(n)
    ppg = np.zeros(n)
    for i, b in enumerate(beats):
        rr = np.diff(beats)[min(i, len(beats) - 2)]
        m = (t > b - 0.4) & (t < b + 0.7)
        ecg[m] += synth_beat(t[m] - b, rr)
        # PPG pulse: systolic + dicrotic Gaussian, arriving pat seconds after R
        m2 = (t > b) & (t < b + 1.2)
        tp = t[m2] - b - pat
        ppg[m2] += np.exp(-0.5 * ((tp - 0.12) / 0.06) ** 2) + 0.35 * np.exp(-0.5 * ((tp - 0.35) / 0.08) ** 2)
    ecg += noise * rng.standard_normal(n) + 0.05 * np.sin(2 * np.pi * 0.25 * t)
    ppg += noise * rng.standard_normal(n) + 0.1 * np.sin(2 * np.pi * 0.2 * t)
    return Record("synthetic", subject, ppg, fs, ecg, fs, ppg_site="sim", meta={"r_times": beats.tolist(), "pat": pat})
