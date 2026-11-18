"""Signal checks on the converted 128 Hz signal: beats, flat line, clipping, polarity, length."""
from __future__ import annotations

import numpy as np

from bench import dsp
from bench.metrics import reference_inverted

from .. import config
from .notices import IntakeError, MODAL, Notice, POPUP, TOAST, TooLong


def check_length(seconds, kind="ppg", limit=config.MAX_SECONDS):
    if seconds < config.MIN_SECONDS:
        what = "PPG" if kind == "ppg" else "true ECG"
        raise IntakeError("File too short",
                          f"This file is only {seconds:.1f} s long. The models need at least "
                          f"{config.MIN_SECONDS:.0f} s of {what}. Please choose a longer file.")
    if limit and seconds > limit + 0.5:
        raise TooLong(seconds, limit)


def clipped_fraction(x) -> float:
    """Share of samples sitting on the signal's extreme values (sensor saturation)."""
    x = np.asarray(x, float)
    if len(x) < 10:
        return 0.0
    lo, hi = x.min(), x.max()
    tol = 1e-6 * max(hi - lo, 1e-12)
    return float(max(np.mean(x <= lo + tol), np.mean(x >= hi - tol)))


def slope_ratio(x, fs) -> float:
    """Steepest rise / steepest fall (98th / 2nd percentile of the derivative of the 0.5-8 Hz signal).
    An upright PPG rises fast and falls slowly (about 1.6-2.1 on BIDMC); a raw absorbance trace with the
    pulse pointing down gives the inverse. Percentiles keep single artefacts from deciding."""
    y = dsp.bandpass(np.asarray(x, float), fs, 0.5, 8.0, order=2)
    d = np.diff(y)
    lo = -np.percentile(d, 2)
    return float(np.percentile(d, 98) / lo) if lo > 0 else float("nan")


def ppg_upside_down(x, fs) -> bool:
    """Only clear cases are flipped (ratio below 0.7, i.e. falls at least 1.4x steeper than rises)."""
    r = slope_ratio(x, fs)
    return bool(np.isfinite(r) and r < 0.7)


def periodicity(x, fs, peaks) -> float:
    """Autocorrelation of the 0.5-8 Hz signal at the median beat interval: about 0.5-1 for a pulse wave,
    near 0 for noise (whose 'peaks' the detector still finds)."""
    if len(peaks) < 3:
        return 0.0
    lag = int(np.median(np.diff(peaks)))
    y = dsp.bandpass(np.asarray(x, float), fs, 0.5, 8.0, order=2)
    if lag < 1 or lag >= len(y) // 2:
        return 0.0
    best = 0.0
    for L in range(max(1, int(lag * 0.9)), int(lag * 1.1) + 2):      # HRV moves the best lag a little
        a, b = y[:-L], y[L:]
        den = np.sqrt(np.sum(a * a) * np.sum(b * b))
        if den > 0:
            best = max(best, float(np.sum(a * b) / den))
    return best


def ppg_checks(x, fs=config.WORK_FS):
    """-> (signal (flipped if upside down), peaks, notices)."""
    notices = []
    x = np.asarray(x, float)
    if np.ptp(x) < 1e-9 or np.std(x) < 1e-9 * max(1.0, abs(np.mean(x))):
        notices.append(Notice(POPUP, "Flat signal", "The PPG is a flat line, so there are no heartbeats to "
                                                    "reconstruct. Is this the right file and column?"))
        return x, np.array([], int), notices
    clip = clipped_fraction(x)
    if clip > 0.05:
        notices.append(Notice(TOAST, "Clipped signal", f"{clip:.0%} of the PPG samples sit at the sensor's limit "
                                                       "(clipped); beats in those parts may be missed."))
    if ppg_upside_down(x, fs):
        x = -x
        notices.append(Notice(TOAST, "PPG flipped", "The PPG looked upside down (pulses pointing down), so it was "
                                                    "flipped before the models saw it."))
    peaks = dsp.detect_ppg_peaks(x, fs)
    secs = len(x) / fs
    if len(peaks) >= 2 and periodicity(x, fs, peaks) < 0.15:
        peaks = np.array([], int)                      # what the detector found in noise are not beats
    if len(peaks) < 2:
        notices.append(Notice(POPUP, "No heartbeats found", "No heartbeats were found in this file. Is it a PPG "
                                                            "signal, and is the sampling rate right? The models "
                                                            "can still run."))
    elif len(peaks) < config.FEW_BEATS or secs < config.FEW_SECONDS:
        notices.append(Notice(TOAST, "Few beats", f"Only {len(peaks)} beats found; measurements will be rough."))
    return x, peaks, notices


def ecg_checks(x, fs=config.WORK_FS):
    """-> (R peaks, inverted?, notices). The true ECG is never altered for scoring (the scorer corrects the
    reference polarity itself); `inverted` is used to draw it upright."""
    notices = []
    x = np.asarray(x, float)
    if np.ptp(x) < 1e-9:
        notices.append(Notice(POPUP, "Flat true ECG", "The true ECG is a flat line; errors cannot be measured "
                                                      "against it."))
        return np.array([], int), False, notices
    y = dsp.bandpass(x, fs, 0.5, 40.0, order=2)
    r = dsp.detect_rpeaks(y, fs)
    inv = reference_inverted(y, r, fs)
    if len(r) < 2:
        notices.append(Notice(POPUP, "No R peaks found", "No heartbeats were found in the true ECG. Is it an ECG, "
                                                         "and is the sampling rate right?"))
    elif inv:
        notices.append(Notice(TOAST, "True ECG upside down", "The true ECG's QRS points down (a reversed or "
                                                             "non-standard lead). It is drawn upright and the "
                                                             "scorer corrects for it, as in Track 1."))
    return r, inv, notices


def blocking(notices):
    return [n for n in notices if n.level == MODAL]
