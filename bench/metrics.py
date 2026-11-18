"""Evaluation metrics (objective D).

Four families, so that "gets the rhythm right" is never confused with
"reproduces the ECG":

  timing-free   hr_mae, rpeak sensitivity / ppv / f1, rpeak timing error, RMSSD error;
                r_f1_dc: F1 after removing the record's constant ECG-to-pulse delay (r_offset_ms)
  waveform-raw  rmse, mae, pearson, prd, fd, dtw          (sample-by-sample, as most papers report)
  waveform-lag  rmse_lag, pearson_lag, best_lag_ms        (best global shift within +-max_lag per window)
  morphology    beat_pearson, beat_rmse                   (matched beats, each centred on its own R peak)

Conventions (all configurable, all recorded in the results):
  * both signals are band-passed with the SAME filter (eval_band) before scoring;
  * waveform metrics use per-window min-max to [-1, 1] (CardioGAN / RDDM convention);
  * R peaks come from the same detector for pred and true (dsp.detect_rpeaks);
  * a REFERENCE whose QRS points down is flagged (ref_inverted = 1); pearson_lag_pc / beat_pearson_pc score
    against the flipped (upright) reference, the plain metrics stay as they were. Beat morphology uses beats
    matched after the delay correction;
  * FD uses curves (sample index, amplitude), matching how values ~3-7 are reported in RDDM.
"""
from __future__ import annotations

import numpy as np

from . import dsp


# ------------------------------------------------------------------ DP distances
def _antidiag_dp(a, b, mode):
    """Discrete Frechet ('max') or DTW ('sum') with vectorised anti-diagonals."""
    n, m = len(a), len(b)
    if mode == "frechet":  # 2-D curves (index, value)
        ia, ib = np.arange(n, dtype=float), np.arange(m, dtype=float)
        d = np.sqrt((ia[:, None] - ib[None, :]) ** 2 + (a[:, None] - b[None, :]) ** 2)
    else:
        d = np.abs(a[:, None] - b[None, :])
    C = np.full((n, m), np.inf)
    for k in range(n + m - 1):
        i = np.arange(max(0, k - m + 1), min(n, k + 1))
        j = k - i
        if k == 0:
            C[0, 0] = d[0, 0]
            continue
        up = np.where(i > 0, C[np.maximum(i - 1, 0), j], np.inf)
        left = np.where(j > 0, C[i, np.maximum(j - 1, 0)], np.inf)
        diag = np.where((i > 0) & (j > 0), C[np.maximum(i - 1, 0), np.maximum(j - 1, 0)], np.inf)
        best = np.minimum(np.minimum(up, left), diag)
        C[i, j] = np.maximum(best, d[i, j]) if mode == "frechet" else best + d[i, j]
    return C[-1, -1]


def frechet(a, b):
    return float(_antidiag_dp(np.asarray(a, float), np.asarray(b, float), "frechet"))


def dtw(a, b, band=None):
    return float(_antidiag_dp(np.asarray(a, float), np.asarray(b, float), "dtw") / len(a))


# ------------------------------------------------------------------ helpers
def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def match_peaks(ref, est, tol):
    """Greedy one-to-one matching within tol samples. Returns (pairs, fn, fp)."""
    ref, est = np.asarray(ref), np.asarray(est)
    used = np.zeros(len(est), bool)
    pairs = []
    for r in ref:
        if len(est) == 0:
            break
        d = np.abs(est - r).astype(float)
        d[used] = np.inf
        k = int(np.argmin(d))
        if d[k] <= tol:
            used[k] = True
            pairs.append((r, est[k]))
    return pairs, len(ref) - len(pairs), int((~used).sum())


def best_lag(a, b, max_lag):
    """Lag (samples) that maximises corr(a shifted, b); positive = a is late."""
    best, best_r = 0, -np.inf
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            x, y = a[L:], b[:len(b) - L]
        else:
            x, y = a[:L], b[-L:]
        r = _pearson(x, y)
        if r > best_r:
            best, best_r = L, r
    return best


METRIC_VERSION = 2   # v2: reference-polarity correction, delay-corrected R-peak matching (Track 1.1)


def reference_inverted(x, rpeaks, fs):
    """True when the reference ECG's QRS points down (e.g. a reversed or non-standard lead, or a deep-S beat).

    Around each detected beat, compare the largest upward and downward excursion from the local median
    (+-60 ms). Aurora-BP's reference is inverted for ~57% of people, which made every model's beat
    correlation there ~0 or negative. Only the REFERENCE is corrected; a generated ECG that is upside down
    is still scored as wrong."""
    if len(rpeaks) < 3:
        return False
    h = max(1, int(0.06 * fs))
    up, down = [], []
    for r in rpeaks:
        seg = x[max(0, r - h): r + h + 1]
        if len(seg) < 3:
            continue
        m = np.median(seg)
        up.append(seg.max() - m)
        down.append(m - seg.min())
    return bool(np.median(down) > 1.2 * np.median(up)) if up else False


def estimate_offset(ref, est, fs, max_s=0.4):
    """Record-level constant delay (samples) of est relative to ref: median of nearest-peak differences
    within +-max_s. A generated ECG that follows the PPG is late by roughly the pulse arrival time; this
    removes that constant without forgiving beat-to-beat jitter."""
    ref, est = np.asarray(ref), np.asarray(est)
    if len(ref) < 5 or len(est) < 5:
        return 0
    k = np.clip(np.searchsorted(est, ref), 1, len(est) - 1)
    d = np.where(np.abs(est[k] - ref) < np.abs(est[k - 1] - ref), est[k] - ref, est[k - 1] - ref)
    d = d[np.abs(d) <= max_s * fs]
    return int(np.round(np.median(d))) if len(d) >= 5 else 0


# ------------------------------------------------------------------ main entry
def evaluate(pred, true, fs, window_s=10.0, eval_band=(0.5, 40.0), max_lag_s=0.3, rpeak_tol_s=0.15,
             compute_fd=True, compute_dtw=False, fd_max_windows=40):
    """Score one continuous prediction against its reference.

    Returns (window_rows, record_row): a list of per-window dicts and one dict
    of record-level metrics (R-peak detection totals, HRV).
    """
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    if eval_band:
        pred = dsp.bandpass(pred, fs, *eval_band, order=2)
        true = dsp.bandpass(true, fs, *eval_band, order=2)

    rp = dsp.detect_rpeaks(pred, fs)
    rt = dsp.detect_rpeaks(true, fs)
    inverted = reference_inverted(true, rt, fs)
    sgn = -1.0 if inverted else 1.0   # *_pc metrics: against the polarity-corrected (upright) reference
    tol = int(round(rpeak_tol_s * fs))
    pairs, fn, fp = match_peaks(rt, rp, tol)
    errs_ms = np.array([(e - r) / fs * 1000 for r, e in pairs]) if pairs else np.array([])
    # delay-corrected matching: remove the record's constant delay, then match within the same tolerance
    off = estimate_offset(rt, rp, fs)
    pairs_dc, fn_dc, fp_dc = match_peaks(rt, rp - off, tol)
    pairs_dc = [(r, e + off) for r, e in pairs_dc]
    errs_dc_ms = np.array([(e - off - r) / fs * 1000 for r, e in pairs_dc]) if pairs_dc else np.array([])

    L = int(round(window_s * fs))
    max_lag = int(round(max_lag_s * fs))
    half_pre, half_post = int(0.25 * fs), int(0.45 * fs)
    rows = []
    n_win = max(1, (len(true) - L) // L + 1)
    fd_stride = 1 if not fd_max_windows else max(1, int(np.ceil(n_win / fd_max_windows)))
    for wi, w0 in enumerate(range(0, len(true) - L + 1, L)):
        sl = slice(w0, w0 + L)
        p = dsp.minmax(pred[sl])
        t = dsp.minmax(true[sl])
        row = dict(t_start=w0 / fs)

        # timing-free
        rt_w = rt[(rt >= w0) & (rt < w0 + L)]
        rp_w = rp[(rp >= w0) & (rp < w0 + L)]
        hr_t = 60.0 * fs / np.mean(np.diff(rt_w)) if len(rt_w) > 1 else np.nan
        hr_p = 60.0 * fs / np.mean(np.diff(rp_w)) if len(rp_w) > 1 else np.nan
        row.update(hr_true=hr_t, hr_pred=hr_p, hr_ae=abs(hr_p - hr_t))
        pw, fnw, fpw = match_peaks(rt_w, rp_w, tol)
        row.update(r_tp=len(pw), r_fn=fnw, r_fp=fpw,
                   r_timing_mae_ms=float(np.mean([abs(e - r) for r, e in pw]) / fs * 1000) if pw else np.nan)
        # beats for morphology: matched after removing the record's constant delay
        pw_dc = [(r, e) for r, e in pairs_dc if w0 <= r < w0 + L]

        # waveform raw
        diff = p - t
        row.update(rmse=float(np.sqrt(np.mean(diff ** 2))), mae=float(np.mean(np.abs(diff))),
                   pearson=_pearson(p, t),
                   prd=float(100 * np.sqrt(np.sum(diff ** 2) / max(np.sum(t ** 2), 1e-12))))
        # FD/DTW are O(L^2): evaluated on evenly spaced windows (<= fd_max_windows per record)
        if compute_fd and wi % fd_stride == 0:
            row["fd"] = frechet(p, t)
        if compute_dtw and wi % fd_stride == 0:
            row["dtw"] = dtw(p, t)

        # waveform after best shift
        lag = best_lag(p, t, max_lag)
        if lag >= 0:
            x, y = p[lag:], t[:L - lag]
        else:
            x, y = p[:lag], t[-lag:]
        row.update(best_lag_ms=lag / fs * 1000, rmse_lag=float(np.sqrt(np.mean((x - y) ** 2))),
                   pearson_lag=_pearson(x, y))
        if inverted:
            tf = -t
            lag2 = best_lag(p, tf, max_lag)
            x2, y2 = (p[lag2:], tf[:L - lag2]) if lag2 >= 0 else (p[:lag2], tf[-lag2:])
            row["pearson_lag_pc"] = _pearson(x2, y2)
        else:
            row["pearson_lag_pc"] = row["pearson_lag"]

        # morphology: matched beats, each centred on its own R peak
        bc, br = [], []
        for r, e in pw_dc:
            if r - half_pre < 0 or e - half_pre < 0 or r + half_post > len(true) or e + half_post > len(pred):
                continue
            bt = dsp.minmax(true[r - half_pre:r + half_post])
            bp = dsp.minmax(pred[e - half_pre:e + half_post])
            bc.append(_pearson(bp, bt))
            br.append(float(np.sqrt(np.mean((bp - bt) ** 2))))
        row.update(beat_pearson=float(np.mean(bc)) if bc else np.nan,
                   beat_rmse=float(np.mean(br)) if br else np.nan,
                   beat_pearson_pc=sgn * float(np.mean(bc)) if bc else np.nan)   # correlation flips with the sign
        rows.append(row)

    # record level
    tp = len(pairs)
    sens = tp / max(tp + fn, 1)
    ppv = tp / max(tp + fp, 1)
    rr_t = np.diff(rt) / fs * 1000
    rr_p = np.diff(rp) / fs * 1000

    def rmssd(rr):
        rr = rr[(rr > 300) & (rr < 2000)]
        return float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 2 else np.nan

    rec = dict(r_sens=sens, r_ppv=ppv, r_f1=2 * sens * ppv / max(sens + ppv, 1e-12),
               r_timing_median_ms=float(np.median(errs_ms)) if len(errs_ms) else np.nan,
               r_timing_iqr_ms=float(np.subtract(*np.percentile(errs_ms, [75, 25]))) if len(errs_ms) else np.nan,
               rmssd_true=rmssd(rr_t), rmssd_pred=rmssd(rr_p), ref_inverted=float(inverted),
               metric_version=METRIC_VERSION)
    tp_dc = len(pairs_dc)
    sens_dc, ppv_dc = tp_dc / max(tp_dc + fn_dc, 1), tp_dc / max(tp_dc + fp_dc, 1)
    rec.update(r_offset_ms=off / fs * 1000, r_sens_dc=sens_dc, r_ppv_dc=ppv_dc,
               r_f1_dc=2 * sens_dc * ppv_dc / max(sens_dc + ppv_dc, 1e-12),
               r_timing_iqr_dc_ms=float(np.subtract(*np.percentile(errs_dc_ms, [75, 25]))) if len(errs_dc_ms) else np.nan)
    rec["rmssd_ae"] = abs(rec["rmssd_pred"] - rec["rmssd_true"])
    return rows, rec
