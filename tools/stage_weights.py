"""Prepare our trained weights (Kaggle jobs k90, k91, k92m, k93) as GitHub Release assets.

    .venv\\Scripts\\python tools\\stage_weights.py --model chiu2020 --weights PATH\\weights.pth --meta PATH\\meta.json

Keeps only what inference needs (the generator / network state, no epoch counters), stores it as fp16 when
the output on a synthetic PPG stays within 1e-3 of fp32 (else fp32; --fp32 forces it), and writes
dist/release/<id>_<job>.zip (weights.pth + meta.json) with its SHA-256. tests/test_track1.py then has the
final word: P2E-WGAN is staged with --fp32 because fp16 moved its Track 1 numbers on BIDMC 05 (its R peaks
are borderline), although the raw output stayed within 1e-3. dist/ is never committed; the zip
is uploaded to the Release by hand and its SHA-256 goes into models/<id>/model.yaml.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

KEEP = {"chiu2020": ["net"], "clep_gan": ["GP", "GE"], "p2e_wgan": ["generator"], "yalcin2026": ["generator"]}
FP16_TOL = 1e-3


def test_windows(ad, seconds=60.0):
    from bench import dsp
    from bench.datasets import make_synthetic
    r = make_synthetic(subject="stage", duration=seconds, fs=250.0, hr=70.0, seed=11)
    x = ad.prepare_record(dsp.resample(r.ppg, r.ppg_fs, ad.native_fs))
    n = len(x) // ad.window_len
    return ad.normalize_window(x[: n * ad.window_len].reshape(n, ad.window_len))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(KEEP))
    ap.add_argument("--weights", required=True, help="weights.pth from the Kaggle job output")
    ap.add_argument("--meta", required=True, help="meta.json from the same job")
    ap.add_argument("--fp32", action="store_true", help="store fp32 even if fp16 passes the 1e-3 check")
    ap.add_argument("--out", default=str(ROOT / "dist" / "release"))
    a = ap.parse_args(argv)

    import torch

    import setup
    from bench.models.trained import TRAINED
    from studio import engine
    spec = {s.id: s for s in engine.discover()}[a.model]
    asset = spec.weights[0]["file"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        repo = setup.fetch_repo(spec, dest=t / "repo")
        st = torch.load(a.weights, map_location="cpu", weights_only=False)
        st = {k: st[k] for k in KEEP[a.model]}
        half = {k: {n: (v.half() if v.is_floating_point() else v) for n, v in sd.items()} for k, sd in st.items()}
        outs = {}
        for name, state in (("fp32", st), ("fp16", half)):
            d = t / name
            d.mkdir()
            shutil.copy2(a.meta, d / "meta.json")
            torch.save(state, d / "weights.pth")
            ad = TRAINED[a.model](str(d), device="cpu")
            ad.repo = repo
            ad.load()
            outs[name] = ad.infer(test_windows(ad))
        diff = float(np.max(np.abs(outs["fp16"] - outs["fp32"])))
        pick = "fp16" if diff <= FP16_TOL and not a.fp32 else "fp32"
        z = out / asset
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
            f.write(t / pick / "weights.pth", "weights.pth")
            f.write(t / pick / "meta.json", "meta.json")
    print(f"{a.model}: max |fp16 - fp32| = {diff:.1e} -> stored as {pick}")
    print(f"  {z}  {z.stat().st_size / 1e6:.1f} MB  sha256 {setup.sha256(z)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
