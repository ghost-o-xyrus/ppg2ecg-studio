"""Zhu DCT adapter (bench.models.zhu_dct, unchanged from Track 1), coefficients from weights/zhu_dct/."""
import numpy as np

from bench.models.zhu_dct import ZhuDCT


def build(ctx):
    p = ctx.spec.params
    m = ZhuDCT(n_ppg=int(p.get("n_ppg", 12)), n_ecg=int(p.get("n_ecg", 100)), alpha=float(p.get("alpha", 500)))
    with np.load(ctx.weights_dir / "zhu_capnobase.npz", allow_pickle=False) as z:
        m.W, m.b, m.pat = z["W"].astype(float), z["b"].astype(float), float(z["pat"])
        ctx.info["fit_on"] = str(z["fit_on"]) if "fit_on" in z.files else "capnobase"
        ctx.info["n_cycles"] = int(z["n_cycles"]) if "n_cycles" in z.files else None
    if m.W.shape != (m.n_ppg, m.n_ecg):
        raise ValueError(f"zhu_capnobase.npz: W has shape {m.W.shape}, expected {(m.n_ppg, m.n_ecg)}")
    m.fitted = True
    return m
