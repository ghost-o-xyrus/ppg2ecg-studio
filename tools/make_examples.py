"""Build the example recordings in examples/ from BIDMC and the UCI Cuff-Less Blood Pressure dataset.

    .venv\\Scripts\\python tools\\make_examples.py --bidmc PATH\\bidmc_csv --uci PATH\\Part_1.mat

BIDMC (PhysioNet, Open Data Commons Attribution 1.0) and the UCI Cuff-Less Blood Pressure Estimation dataset
(MIMIC-II recordings, CC BY 4.0) allow redistribution with attribution; examples/README.md gives it. Each file
is a cut of one recording, written in one of the layouts the app reads, so the files also exercise the intake
(columns, sampling rate from a time column / comment / file name, windows, gaps, polarity, length limits).
The only changes to the values are the ones listed per file (a sign flip, a gap of missing values, a
resampling); everything else is the recording as published.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

FS = 125.0


def bidmc(folder, subject):
    import pandas as pd
    d = pd.read_csv(Path(folder) / f"bidmc_{subject:02d}_Signals.csv")
    d.columns = [c.strip() for c in d.columns]
    return d


def cut(x, seconds, start=0.0):
    return np.array(x[int(start * FS): int((start + seconds) * FS)], float)          # a copy


def write_csv(path, cols, header, comment=None, fmt="%.5f"):
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        if comment:
            f.write(comment + "\n")
        f.write(",".join(header) + "\n")
        np.savetxt(f, np.column_stack(cols), delimiter=",", fmt=fmt)


def uci_records(path, n=400):
    import h5py
    with h5py.File(path, "r") as f:
        refs = f[next(k for k in f.keys() if not k.startswith("#"))][:].ravel()[:n]
        for i, r in enumerate(refs):
            yield i, np.asarray(f[r]).T if np.asarray(f[r]).shape[0] == 3 else np.asarray(f[r])


def pick_uci(path, seconds, avoid=()):
    """First record at least `seconds` long whose PPG and ECG both show a clean, matching rhythm."""
    from bench import dsp
    for i, rec in uci_records(path):
        if i in avoid or len(rec) < seconds * FS:
            continue
        ppg, ecg = rec[: int(seconds * FS), 0], rec[: int(seconds * FS), 2]
        if not (np.isfinite(ppg).all() and np.isfinite(ecg).all()) or np.ptp(ppg) == 0 or np.ptp(ecg) == 0:
            continue
        pk = dsp.detect_ppg_peaks(dsp.resample(ppg, FS, 128), 128)
        rp = dsp.detect_rpeaks(dsp.bandpass(dsp.resample(ecg, FS, 128), 128, 0.5, 40, order=2), 128)
        bpm = 60 * len(rp) / seconds
        if 50 <= bpm <= 120 and abs(len(pk) - len(rp)) <= 0.03 * len(rp):
            return i, rec[: int(seconds * FS)]
    raise SystemExit(f"no clean UCI record of {seconds:.0f} s found")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bidmc", required=True, help="folder with bidmc_XX_Signals.csv")
    ap.add_argument("--uci", required=True, help="Part_1.mat of the UCI Cuff-Less Blood Pressure dataset")
    ap.add_argument("--out", default=str(ROOT / "examples"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = a.bidmc

    # 1. the original BIDMC layout, whole 8 min (the app offers its first 2 min; ECG from column II)
    src = Path(B) / "bidmc_25_Signals.csv"
    (out / "bidmc_25_full_8min.csv").write_bytes(src.read_bytes().replace(b"\r\n", b"\n"))

    # 2-3. PPG and true ECG as separate files; the ECG file starts 3 s later (aligned by the time columns)
    d = bidmc(B, 13)
    t = np.arange(int(120 * FS)) / FS
    write_csv(out / "bidmc_13_ppg_2min.csv", [t, cut(d["PLETH"], 120)], ["time", "ppg"], fmt="%.4f")
    write_csv(out / "bidmc_13_ecg_2min.csv", [t[: int(117 * FS)] + 3.0, cut(d["II"], 117, 3.0)], ["time", "ecg"],
              fmt="%.4f")

    # 4. one bare column, rate in the file name
    np.savetxt(out / "bidmc_21_ppg_60s_125Hz.txt", cut(bidmc(B, 21)["PLETH"], 60), fmt="%.5f")

    # 5. rate in a comment line
    write_csv(out / "bidmc_19_ppg_30s.csv", [cut(bidmc(B, 19)["PLETH"], 30)], ["ppg"], comment="# fs=125")

    # 6. short: about 19 beats, measurements rough
    d = bidmc(B, 9)
    write_csv(out / "bidmc_09_ppg_15s.csv", [np.arange(int(15 * FS)) / FS, cut(d["PLETH"], 15)], ["time", "ppg"],
              fmt="%.4f")

    # 7. too short (8 s < 10 s)
    d = bidmc(B, 22)
    write_csv(out / "bidmc_22_ppg_8s.csv", [np.arange(int(8 * FS)) / FS, cut(d["PLETH"], 8)], ["time", "ppg"],
              fmt="%.4f")

    # 8. NumPy archive with PPG, ECG and the rate
    d = bidmc(B, 45)
    np.savez(out / "bidmc_45_ppg_ecg_90s.npz", ppg=cut(d["PLETH"], 90), ecg=cut(d["II"], 90), fs=np.float64(FS))

    # 9. NumPy array, rate in the file name
    np.save(out / "bidmc_08_ppg_90s_125Hz.npy", cut(bidmc(B, 8)["PLETH"], 90))

    # 10. rows of 4-s windows (2 min), rate in the file name
    np.save(out / "bidmc_16_ppg_windows_4s_125Hz.npy", cut(bidmc(B, 16)["PLETH"], 120).reshape(30, int(4 * FS)))

    # 11. PPG stored upside down (sign flipped here); the app flips it back
    d = bidmc(B, 27)
    write_csv(out / "bidmc_27_ppg_upside_down_60s.csv", [np.arange(int(60 * FS)) / FS, -cut(d["PLETH"], 60)],
              ["time", "ppg"], fmt="%.4f")

    # 12. time in milliseconds, a 0.6-s gap of missing PPG values, and an ECG recorded upside down
    d = bidmc(B, 3)
    ppg = cut(d["PLETH"], 120)
    ppg[int(50 * FS): int(50.6 * FS)] = np.nan
    write_csv(out / "bidmc_03_ppg_ecg_gap_2min.csv", [np.arange(len(ppg)) / FS * 1000.0, ppg, cut(d["II"], 120)],
              ["time_ms", "ppg", "ecg"], fmt="%.4f")

    # 13. a hard case: many beats leave no clear pulse in the PPG (pulse deficit)
    d = bidmc(B, 41)
    write_csv(out / "bidmc_41_ppg_ecg_2min.csv", [np.arange(int(120 * FS)) / FS, cut(d["PLETH"], 120),
                                                  cut(d["II"], 120)], ["time", "ppg", "ecg"], fmt="%.4f")

    # 14-15. UCI Cuff-Less Blood Pressure (MIMIC-II): PPG, arterial pressure, ECG
    i, rec = pick_uci(a.uci, 120)
    write_csv(out / "mimic2_ppg_abp_ecg_2min.csv", [np.arange(len(rec)) / FS, rec[:, 0], rec[:, 1], rec[:, 2]],
              ["time", "ppg", "abp", "ecg"], fmt="%.4f")
    j, rec = pick_uci(a.uci, 75, avoid=(i,))
    from bench import dsp
    ppg, ecg = dsp.resample(rec[:, 0], FS, 250.0), dsp.resample(rec[:, 2], FS, 250.0)
    write_csv(out / "mimic2_ppg_ecg_75s_250Hz.csv", [np.arange(len(ppg)) / 250.0, ppg, ecg], ["time", "ppg", "ecg"],
              fmt="%.4f")
    print(f"UCI Part_1 records used: {i} (2 min), {j} (75 s)")
    for f in sorted(out.iterdir()):
        if f.suffix != ".md":
            print(f"  {f.name:<38} {f.stat().st_size / 1e3:8.0f} kB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
