"""Command-line run: the same intake, models and scorer as the window, printed as tables.

    python -m studio.cli PPG_FILE [--ecg ECG_FILE | --ecg same] [--models template,clep_gan] [--rate 125]
                         [--column PLETH] [--full] [--out reconstruction_folder]

--ecg same   take the true ECG from the PPG file itself (e.g. BIDMC: PLETH and II in one CSV)
--full       no 2-minute limit (score whole recordings, as in Track 1)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import config, engine, intake, scoring


def table(columns, results, title):
    names = [r.model_name for r in results]
    w0 = max(len("model"), *(len(n) for n in names))
    rows = [r.row(columns) for r in results]
    cells = [[scoring.fmt(v, c) for v, c in zip(row, columns)] for row in rows]
    widths = [max(len(c.label), *(len(x[i]) for x in cells)) for i, c in enumerate(columns)]
    lines = [title, "  ".join(["model".ljust(w0)] + [c.label.rjust(w) for c, w in zip(columns, widths)])]
    for n, x in zip(names, cells):
        lines.append("  ".join([n.ljust(w0)] + [v.rjust(w) for v, w in zip(x, widths)]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m studio.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ppg")
    ap.add_argument("--ecg")
    ap.add_argument("--models", default="")
    ap.add_argument("--rate", type=float)
    ap.add_argument("--ecg-rate", type=float)
    ap.add_argument("--column")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    limit = None if a.full else config.MAX_SECONDS
    mb = None if a.full else config.MAX_MB
    settings = config.Settings()
    try:
        p = intake.prepare(a.ppg, "ppg", settings, rate=a.rate, column=a.column, limit_s=limit, limit_mb=mb,
                           keep_first=True)
        e = None
        if a.ecg == "same":
            if not p.embedded:
                sys.exit("no ECG column found in the PPG file")
            e = intake.prepare_embedded(p, next(iter(p.embedded)), limit_s=limit, keep_first=True)
        elif a.ecg:
            e = intake.prepare(a.ecg, "ecg", settings, rate=a.ecg_rate or a.rate, limit_s=limit, limit_mb=mb,
                               keep_first=True)
        ppg, ecg, notes = intake.align(p, e)
    except intake.NeedRate:
        sys.exit("The sampling rate is not in the file: pass --rate HZ")
    except intake.NeedColumn as ex:
        sys.exit(f"Several columns could be the signal: pass --column one of {ex.candidates}")
    except intake.IntakeError as ex:
        sys.exit(f"{ex.title}: {ex.text}")
    print(f"PPG  {p.name}: {p.seconds:.1f} s, {p.fs_in:g} Hz ({p.fs_source}) -> 128 Hz, column {p.column}, "
          f"{p.n_beats} beats")
    if e is not None:
        print(f"ECG  {e.name}: {e.seconds:.1f} s, {e.fs_in:g} Hz -> 128 Hz, {e.n_beats} R peaks")
    for n in p.notices + (e.notices if e else []) + notes:
        print("  note:", n.title, "-", n.text)
    want = [m.strip() for m in a.models.split(",") if m.strip()]
    specs = [s for s in engine.discover() if not want or s.id in want]
    from .engine import device as devmod
    dev = devmod.probe(settings.get("device", "auto"))
    results = []
    peaks = intake.aligned_peaks(p, ppg)
    src = intake.model_input(p, e, ppg)
    for s in specs:
        lm = engine.load_model(s, dev)
        if lm.status != engine.READY:
            print(f"  {s.name}: {lm.status} ({lm.error})")
            continue
        r = engine.run_model(lm, ppg, peaks, ecg, src=src)
        results.append(r)
        if a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            np.savetxt(Path(a.out) / f"{Path(a.ppg).stem}_{s.id}_128Hz.csv",
                       np.c_[np.arange(len(r.pred)) / config.WORK_FS, r.pred], delimiter=",",
                       header="time_s,ecg", comments="", fmt="%.6f")
    if not results:
        sys.exit("no model could run")
    print()
    print(table(scoring.COMPARISON, results, "Model comparison"))
    print()
    if ecg is not None:
        print(table(scoring.ERRORS, results, "Errors vs true ECG (Track 1 scorer)"))
    else:
        print("Errors vs true ECG: no true ECG was given.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
