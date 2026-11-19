"""Engine: model registry, device choice, reconstruction and scoring of one entry.

v0.1 runs in-process on worker threads; the API below is what the separate engine process will expose in
v0.2, so the window code does not change when it moves out.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from .. import config, scoring

os.environ.setdefault("TQDM_DISABLE", "1")      # PPGFlowECG's sampler would print a progress bar
from .registry import FAILED, LOADING, MISSING, READY, LoadedModel, ModelSpec, discover, load_model  # noqa: F401


@dataclass
class Result:
    model_id: str
    model_name: str
    pred: np.ndarray                 # full-length reconstruction at 128 Hz (stitched, as produced)
    gen_s: float                     # wall-clock time of this model's run
    infer_s: float                   # model inference only
    n_windows: int
    device: str
    vs_ppg: dict = field(default_factory=dict)
    vs_true: dict | None = None      # Track 1 summary, when a true ECG was given
    windows: list | None = None      # per-window Track 1 rows
    seed: int | None = None
    error: str = ""

    @property
    def rt_factor(self):
        return self.gen_s / max(len(self.pred) / config.WORK_FS, 1e-9)

    def row(self, columns):
        d = dict(self.vs_ppg)
        if self.vs_true:
            d.update(self.vs_true)
        d.update(gen_s=self.gen_s, rt_factor=self.rt_factor, device=self.device)
        return [d.get(c.key, float("nan")) for c in columns]


def run_order(ids, specs):
    """Fast models first (measured speed from setup, else the manifest order); the slow sampled models last."""
    import json
    try:
        speed = json.loads((config.WEIGHTS_DIR / "speed.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        speed = {}
    by_id = {s.id: s for s in specs}

    def key(i):
        s = by_id.get(i)
        rt = speed.get(i, {}).get("rt_factor")
        return (bool(s and s.stochastic), rt if rt is not None else 1e9, s.order if s else 99)
    return sorted(ids, key=key)


def reconstruct(lm: LoadedModel, ppg128, seed=0, src=None):
    """Whole signal in, whole reconstruction out at 128 Hz (bench.runner.run_record: the model's own windows,
    its preprocessing, overlap 0.5 with tapered stitching, or the stitching its authors require).
    src = (signal, rate) at the file's own rate (intake.model_input); the model is then fed from it, so it is
    resampled once, straight to the model's rate, as in Track 1."""
    from bench.record import Record
    from bench.runner import run_record
    x = np.asarray(ppg128, float)
    sx, sfs = (np.asarray(src[0], float), float(src[1])) if src is not None else (x, config.WORK_FS)
    rec = Record("studio", "entry", sx, sfs, sx, sfs)   # ECG slot unused here
    if lm.spec.stochastic:
        np.random.seed(seed)
        if hasattr(lm.adapter, "seed"):             # RDDM, PPGFlowECG: torch.manual_seed(adapter.seed) per run
            lm.adapter.seed = int(seed)
    with lm.lock:                                   # one run per model at a time
        t0 = time.perf_counter()
        out = run_record(lm.adapter, rec, eval_fs=config.WORK_FS, overlap=config.STITCH_OVERLAP,
                         rng=np.random.default_rng(seed))
        gen = time.perf_counter() - t0
    pred = np.asarray(out.pred, float)
    if len(pred) < len(x):                          # never shorter than the input
        pred = np.pad(pred, (0, len(x) - len(pred)))
    return pred[: len(x)], gen, float(out.infer_seconds), int(out.n_windows)


def run_model(lm: LoadedModel, ppg128, ppg_peaks, ecg128=None, seed=0, src=None) -> Result:
    pred, gen, inf, nw = reconstruct(lm, ppg128, seed, src)
    r = Result(lm.spec.id, lm.spec.name, pred, gen, inf, nw, lm.device,
               seed=seed if lm.spec.stochastic else None)
    r.vs_ppg = scoring.score_vs_ppg(pred, ppg_peaks)
    if ecg128 is not None:
        r.vs_true, r.windows, _ = scoring.score_vs_true(pred, ecg128)
    return r
