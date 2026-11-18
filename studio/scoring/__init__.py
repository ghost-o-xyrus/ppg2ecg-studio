"""Measurements. Errors against a true ECG use the benchmark's own scorer (bench.metrics.evaluate, Track 1
settings), so the numbers mean exactly what they mean in the reports. Without a true ECG, the reconstruction
is compared with the PPG it came from (heart rate, beats drawn vs pulses found)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import config


@dataclass(frozen=True)
class Column:
    key: str
    label: str
    unit: str = ""
    digits: int = 2
    better: str | None = None      # 'low' | 'high' | 'zero' (closest to 0) | None
    tip: str = ""


# ---------------------------------------------------------------- comparison (no true ECG needed)
COMPARISON = [
    Column("hr_recon", "HR, reconstruction", "bpm", 1, None, "Heart rate of the reconstructed ECG (R peaks)."),
    Column("hr_ppg", "HR, PPG pulses", "bpm", 1, None, "Heart rate of the PPG pulses: the reference column."),
    Column("hr_diff", "HR difference", "bpm", 1, "zero", "Reconstruction minus PPG heart rate."),
    Column("beats", "Beats drawn", "", 0, None, "R peaks found in the reconstruction."),
    Column("pulses", "Pulses found", "", 0, None, "Systolic peaks found in the PPG."),
    Column("pulse_f1", "Beats match pulses (F1)", "", 3, "high",
           "Agreement of drawn beats with PPG pulses after removing their one constant delay (1 = every pulse "
           "has exactly one beat)."),
    Column("rmssd", "RMSSD, reconstruction", "ms", 1, None, "Beat-to-beat variability of the reconstruction."),
    Column("rmssd_ppg", "RMSSD, PPG", "ms", 1, None, "Beat-to-beat variability of the PPG pulses."),
    Column("flags", "Flags", "", 0, None, "flat output, missing or extra beats"),
    Column("gen_s", "Generation time", "s", 3, "low", "Wall-clock time of this model's run (windowing, inference, "
                                                     "stitching); not loading, not the shared conversion."),
    Column("rt_factor", "Real-time factor", "", 4, "low", "Generation time / signal length (0.01 = 100x faster "
                                                         "than real time)."),
    Column("device", "Device", "", 0, None, ""),
]

# ---------------------------------------------------------------- errors vs true ECG (Track 1 scorer)
ERRORS = [
    Column("hr_ae", "Heart-rate error", "bpm", 2, "low", "Mean absolute HR error over 10-s windows."),
    Column("r_f1", "R-peak F1 (raw)", "", 3, "high", "R peaks within 150 ms of the true ones."),
    Column("r_f1_dc", "R-peak F1 (delay-corrected)", "", 3, "high",
           "After removing the record's one constant delay (Track 1 headline R-peak metric)."),
    Column("r_offset_ms", "Delay", "ms", 0, None, "Constant delay of the reconstruction's R peaks."),
    Column("r_jitter_ms", "R-peak jitter (IQR)", "ms", 1, "low",
           "Spread of R-peak timing errors after removing the delay."),
    Column("pearson", "Waveform r (raw)", "", 3, "high", "Pearson r per 10-s window, no shift."),
    Column("pearson_lag_pc", "Waveform r (best lag)", "", 3, "high",
           "Pearson r at the best shift within +-300 ms, reference polarity corrected."),
    Column("beat_pearson_pc", "Beat shape r", "", 3, "high",
           "Mean correlation of matched beats, each centred on its own R peak (Track 1 morphology metric)."),
    Column("rmse", "RMSE", "", 3, "low", "Per window, both scaled to [-1, 1]."),
    Column("prd", "PRD", "%", 1, "low", "Percentage root-mean-square difference."),
    Column("fd", "Frechet distance", "", 3, "low", "Discrete Frechet distance per window (up to 20 windows)."),
    Column("rmssd_ae", "RMSSD error", "ms", 1, "low", "|RMSSD of reconstruction - RMSSD of true ECG|."),
    Column("gen_s", "Generation time", "s", 3, "low", ""),
    Column("rt_factor", "Real-time factor", "", 4, "low", ""),
]


def _nanmean(rows, k):
    v = np.array([r.get(k, np.nan) for r in rows], float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def score_vs_true(pred, true, fs=config.WORK_FS):
    """Track 1 scoring of one reconstruction: subject value = mean over 10-s windows, plus the record-level
    values (R-peak detection totals, delay, RMSSD). -> (summary dict, window rows, record dict)."""
    from bench.metrics import evaluate
    rows, rec = evaluate(pred, true, fs, window_s=config.EVAL_WINDOW_S, fd_max_windows=config.FD_MAX_WINDOWS)
    out = {k: _nanmean(rows, k) for k in ("hr_ae", "hr_true", "hr_pred", "pearson", "pearson_lag", "pearson_lag_pc",
                                          "beat_pearson", "beat_pearson_pc", "rmse", "rmse_lag", "prd", "fd",
                                          "r_timing_mae_ms")}
    for k in ("r_f1", "r_f1_dc", "r_sens", "r_ppv", "r_offset_ms", "rmssd_true", "rmssd_pred", "rmssd_ae",
              "ref_inverted"):
        out[k] = float(rec.get(k, np.nan))
    out["r_jitter_ms"] = float(rec.get("r_timing_iqr_dc_ms", np.nan))
    out["n_windows"] = len(rows)
    return out, rows, rec


def _rmssd_ms(peaks, fs):
    rr = np.diff(np.asarray(peaks)) / fs * 1000
    rr = rr[(rr > 300) & (rr < 2000)]
    return float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 2 else float("nan")


def _hr(peaks, fs):
    rr = np.diff(np.asarray(peaks)) / fs
    rr = rr[(rr > 0.28) & (rr < 2.0)]
    return float(60.0 / np.median(rr)) if len(rr) else float("nan")


def recon_rpeaks(pred, fs=config.WORK_FS):
    from bench import dsp
    y = dsp.bandpass(np.asarray(pred, float), fs, 0.5, 40.0, order=2)
    return dsp.detect_rpeaks(y, fs) if np.ptp(y) > 1e-9 else np.array([], int)


def score_vs_ppg(pred, ppg_peaks, fs=config.WORK_FS):
    """Measures that need no true ECG: the reconstruction against the PPG pulses it was made from."""
    from bench.metrics import estimate_offset, match_peaks
    rp = recon_rpeaks(pred, fs)
    pk = np.asarray(ppg_peaks, int)
    out = dict(hr_recon=_hr(rp, fs), hr_ppg=_hr(pk, fs), beats=int(len(rp)), pulses=int(len(pk)),
               rmssd=_rmssd_ms(rp, fs), rmssd_ppg=_rmssd_ms(pk, fs))
    out["hr_diff"] = out["hr_recon"] - out["hr_ppg"]
    if len(rp) >= 2 and len(pk) >= 2:
        off = estimate_offset(rp, pk, fs, max_s=0.6)          # PPG pulses come one arrival time after R
        pairs, fn, fp = match_peaks(rp, pk - off, int(round(0.15 * fs)))
        tp = len(pairs)
        sens, ppv = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
        out["pulse_f1"] = 2 * sens * ppv / max(sens + ppv, 1e-12)
        out["pulse_delay_ms"] = off / fs * 1000
    else:
        out["pulse_f1"], out["pulse_delay_ms"] = 0.0, float("nan")
    flags = []
    if np.ptp(np.asarray(pred, float)) < 1e-9 or len(rp) < 2:
        flags.append("flat")
    elif len(pk) >= 5:
        if len(rp) < 0.8 * len(pk):
            flags.append("missing beats")
        elif len(rp) > 1.2 * len(pk):
            flags.append("extra beats")
    out["flags"] = ", ".join(flags) if flags else "-"
    return out


def display_shift(summary) -> int:
    """Samples to shift a reconstruction earlier so it lines up with the true ECG (its constant delay)."""
    v = summary.get("r_offset_ms", float("nan")) if summary else float("nan")
    return int(round(v / 1000.0 * config.WORK_FS)) if v is not None and math.isfinite(v) else 0


def best_index(values, better):
    """Index of the best finite value for a column, or None."""
    v = np.array([x if isinstance(x, (int, float)) else np.nan for x in values], float)
    ok = np.isfinite(v)
    if better is None or ok.sum() < 2:
        return None
    if better == "low":
        return int(np.nanargmin(np.where(ok, v, np.nan)))
    if better == "high":
        return int(np.nanargmax(np.where(ok, v, np.nan)))
    if better == "zero":
        return int(np.nanargmin(np.where(ok, np.abs(v), np.nan)))
    return None


def fmt(value, col: Column) -> str:
    if isinstance(value, str):
        return value
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    if col.digits == 0:
        return f"{value:.0f}"
    return f"{value:.{col.digits}f}"
