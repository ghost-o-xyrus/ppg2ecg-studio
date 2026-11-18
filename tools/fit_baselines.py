"""Fit the two built-in baselines exactly as Track 1 did, and write their weights for the app.

    python tools/fit_baselines.py --capnobase PATH_TO_CAPNOBASE [--out weights]

* Zhu DCT: ridge regression (lambda 500) from 12 PPG to 100 ECG DCT coefficients per heartbeat cycle, fitted
  once on all 42 CapnoBase recordings -> weights/zhu_dct/zhu_capnobase.npz (W, b, pat, n_cycles).
  Track 1 reference: 27,661 cycles, pat 0.234375 s.
* Template baseline: median R -> PPG-peak delay on CapnoBase -> weights/template/template.json.
  Track 1 reference: pat 0.3828125 s (the value built into models/template/model.yaml).

CapnoBase (IEEE TBME RR benchmark, 42 x 8 min) is read in whichever form it was downloaded:
*_8min_signal.csv, *_8min_signal.tab or *_8min.mat (MATLAB v7.3 files need h5py: pip install h5py).
`python setup.py --zhu --capnobase PATH` runs this for you (and installs h5py into .venv if needed).
--dataset bidmc fits on BIDMC instead (development only; the app marks such weights).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from bench.datasets import load_bidmc, load_capnobase  # noqa: E402
from bench.models.template_baseline import TemplateBaseline  # noqa: E402
from bench.models.zhu_dct import ZhuDCT  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capnobase", help="folder holding the CapnoBase files")
    ap.add_argument("--dataset", choices=["capnobase", "bidmc"], default="capnobase")
    ap.add_argument("--bidmc", help="folder holding bidmc_*_Signals.csv (with --dataset bidmc)")
    ap.add_argument("--out", default=str(ROOT / "weights"))
    a = ap.parse_args(argv)
    root = a.capnobase if a.dataset == "capnobase" else a.bidmc
    if not root:
        ap.error(f"--{a.dataset} PATH is required")
    t0 = time.time()
    loader = load_capnobase if a.dataset == "capnobase" else load_bidmc
    try:
        recs = list(loader(root))
    except ModuleNotFoundError as e:
        sys.exit(f"{e}. CapnoBase .mat files are MATLAB v7.3 (HDF5): run  pip install h5py  and try again.")
    print(f"{len(recs)} {a.dataset} recordings read in {time.time() - t0:.0f} s")
    out = Path(a.out)

    zhu = ZhuDCT()
    info = zhu.fit(recs)
    zd = out / "zhu_dct"
    zd.mkdir(parents=True, exist_ok=True)
    zf = zd / "zhu_capnobase.npz"
    np.savez(zf, W=zhu.W, b=zhu.b, pat=np.float64(zhu.pat), n_cycles=np.int64(info["n_cycles"]),
             fit_on=np.str_(a.dataset), n_records=np.int64(len(recs)))
    print(f"Zhu DCT: {info['n_cycles']} cycles, pat {zhu.pat:.6f} s -> {zf}  sha256 {sha256(zf)}")

    tb = TemplateBaseline()
    pat = tb.fit(recs)
    td = out / "template"
    td.mkdir(parents=True, exist_ok=True)
    tf = td / "template.json"
    tf.write_text(json.dumps({"pat": pat, "fit_on": a.dataset, "n_records": len(recs)}, indent=1))
    print(f"Template: pat {pat:.7f} s -> {tf}")
    if a.dataset == "capnobase":
        ok = info["n_cycles"] == 27661 and abs(zhu.pat - 0.234375) < 1e-9 and abs(pat - 0.3828125) < 1e-9
        print("Matches Track 1 (27,661 cycles, pat 0.234375 / 0.3828125)." if ok else
              "NOTE: differs from Track 1 (27,661 cycles, Zhu pat 0.234375, template pat 0.3828125); "
              "check that all 42 recordings were found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
