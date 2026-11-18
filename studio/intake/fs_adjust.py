"""Sampling-rate step: find the rate, check it is plausible, convert to the 128 Hz working rate.

Rate, in order: what the file states (time column, '# fs=' comment, npz fs array), a '_125Hz' hint in the
file name, the rate remembered for the file's folder, else the user is asked (NeedRate).

Plausibility: at that rate the pulse's main frequency must fall at 36-210 beats per minute. If it does not,
a side notification says the rate looks wrong and shows the heart rate it would imply. Resampling uses the
benchmark's anti-aliased polyphase filter (bench.dsp.resample), exactly as in Track 1.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from bench import dsp

from .. import config
from .notices import NeedRate, Notice, TOAST


def find_rate(raw, path, settings=None, rate=None):
    """-> (fs, source). Raises NeedRate when nothing says what the rate is."""
    if rate:
        return float(rate), "you"
    if raw.fs:
        return float(raw.fs), raw.fs_source
    if settings is not None:
        import os
        r = settings.folder_rate(os.path.dirname(os.path.abspath(path)))
        if r:
            return r, "remembered for this folder"
    raise NeedRate(str(path))


def pulse_rate_bpm(x, fs, kind="ppg") -> float:
    """Main repetition rate of the signal (beats/min) from the spectrum of its pulse envelope.

    PPG: band-limited signal itself. ECG: smoothed energy of the QRS band, whose fundamental is the heart rate
    (the raw ECG spectrum is dominated by QRS harmonics). Searched widely (3-600 bpm) so that a wrong rate
    shows up as an implausible heart rate instead of being missed."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 4 * fs or np.ptp(x) < 1e-12:
        return float("nan")
    nyq = fs / 2
    if kind == "ecg":
        hi = min(30.0, 0.9 * nyq)
        lo = min(5.0, hi / 3)
        e = dsp.bandpass(x, fs, lo, hi, order=2) ** 2
        k = max(1, int(0.1 * fs))
        y = np.convolve(e, np.ones(k) / k, mode="same")
    else:
        y = dsp.bandpass(x, fs, min(0.04, 0.2 * nyq), min(12.0, 0.9 * nyq), order=2)
    y = y - y.mean()
    nper = int(min(len(y), max(8 * fs, 256)))
    f, p = sps.welch(y, fs, nperseg=nper, noverlap=nper // 2, nfft=max(nper, int(2 ** np.ceil(np.log2(nper))) * 2))
    m = (f >= 0.05) & (f <= 10.0)
    if not m.any():
        return float("nan")
    f, p = f[m], p[m]
    i = int(np.argmax(p))
    # a strong sub-harmonic (respiration, baseline wander) can outrank the pulse in raw PPG:
    # prefer the strongest peak inside the plausible band if it is at least a quarter as strong
    lo, hi = config.HR_RANGE_BPM[0] / 60, config.HR_RANGE_BPM[1] / 60
    band = (f >= lo) & (f <= hi)
    if band.any() and not band[i]:
        j = int(np.flatnonzero(band)[np.argmax(p[band])])
        if p[j] >= 0.25 * p[i]:
            i = j
    return float(f[i] * 60.0)


def check_rate(x, fs, kind="ppg"):
    """-> (bpm, notice or None)."""
    bpm = pulse_rate_bpm(x, fs, kind)
    lo, hi = config.HR_RANGE_BPM
    if np.isfinite(bpm) and not (lo <= bpm <= hi):
        what = "PPG" if kind == "ppg" else "true ECG"
        return bpm, Notice(TOAST, "Sampling rate looks wrong",
                           f"At {fs:g} Hz the {what} would beat at {bpm:.0f} per minute (expected {lo:.0f}-{hi:.0f}). "
                           "If the plot looks odd, drop the file again and give the right rate.")
    return bpm, None


def to_work_rate(x, fs):
    return dsp.resample(np.asarray(x, float), float(fs), config.WORK_FS)
