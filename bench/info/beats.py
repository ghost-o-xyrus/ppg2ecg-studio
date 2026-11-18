"""Beat-level pairing of ECG and PPG, block templates, ECG delineation and PPG pulse features.

Everything in bench/info is built on the table this module produces: one row per BLOCK of
`block_beats` consecutive, matched heartbeats, with
  * ECG features of the block's median beat (RR, QRS width, PR, P/T amplitude, ST level, QT, QTc)
  * PPG features of the block's median pulse (timing, morphology), computed WITHOUT the ECG
  * a split-half copy of every ECG feature (odd vs even beats) -> measurement noise ceiling

Median templates over ~10 beats are used because single-beat delineation of P waves and T ends
is unreliable on bedside / wearable data; block templates are the standard compromise.

numpy / scipy only (neurokit2 is not needed), so it runs in Kaggle's base Python and here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps
from scipy.fft import dct

from .. import dsp

FS = 256.0                     # working rate: 3.9 ms resolution for QRS/PR/QT
PRE, POST = 0.35, 0.65         # ECG template window around R (s)
PULSE_N = 64                   # PPG pulse resampled foot-to-foot to this many samples
N_DCT = 12                     # DCT coefficients kept for the pulse shape
ECG_TARGETS = ["rr_ms", "rr_sd_ms", "rr_rmssd_ms", "qrs_ms", "pr_ms", "p_amp", "st_rel", "t_amp",
               "qt_ms", "qtc_f_ms", "r_s_ratio"]


@dataclass
class BeatPairs:
    """Matched heartbeats of one record (indices at FS)."""
    r: np.ndarray            # R peaks
    foot: np.ndarray         # matched PPG pulse onset for each R
    peak: np.ndarray         # matched PPG systolic peak
    ecg: np.ndarray          # filtered ECG at FS
    ppg: np.ndarray          # filtered PPG at FS
    meta: dict = field(default_factory=dict)

    @property
    def pat_ms(self):
        return (self.foot - self.r) / FS * 1000.0


# ------------------------------------------------------------------ pairing
def refine_feet(ppg, peaks, fs=FS):
    """Pulse onset by the intersecting-tangent method (more stable than the minimum for timing):
    the tangent at maximum upslope crosses the horizontal through the preceding minimum."""
    d = np.gradient(ppg)
    feet = []
    for p in peaks:
        a = max(0, p - int(0.4 * fs))
        if p - a < 3:
            feet.append(a)
            continue
        i_min = a + int(np.argmin(ppg[a:p + 1]))
        seg = slice(i_min, p + 1)
        i_up = i_min + int(np.argmax(d[seg]))
        slope = d[i_up]
        if slope <= 0:
            feet.append(i_min)
            continue
        t = i_up - (ppg[i_up] - ppg[i_min]) / slope
        feet.append(int(round(np.clip(t, i_min, i_up))))
    return np.asarray(feet, dtype=int)


def pair_beats(rec, pat_range=(0.05, 0.60), ecg_band=(0.67, 40.0)) -> BeatPairs:
    """Resample to FS, detect R peaks and PPG pulses, and give every R peak the first PPG onset
    that follows it within pat_range seconds (unmatched beats are dropped)."""
    ecg = dsp.resample(rec.ecg, rec.ecg_fs, FS)
    ppg = dsp.resample(rec.ppg, rec.ppg_fs, FS)
    n = min(len(ecg), len(ppg))
    ecg, ppg = ecg[:n], ppg[:n]
    # 0.67 Hz zero-phase high-pass keeps the ST segment usable (AHA allows 0.67 Hz with linear phase)
    ecg = dsp.bandpass(ecg, FS, *ecg_band, order=2)
    ppg = dsp.bandpass(ppg, FS, 0.5, 8.0, order=3)
    r = dsp.detect_rpeaks(ecg, FS)
    # make R positive on average so the template has an upright R
    if len(r) and np.median(ecg[r]) < 0:
        ecg = -ecg
        r = dsp.detect_rpeaks(ecg, FS)
    pk = dsp.detect_ppg_peaks(ppg, FS)
    ft = refine_feet(ppg, pk)
    R, F, P = [], [], []
    j = 0
    for ri in r:
        while j < len(ft) and ft[j] < ri + pat_range[0] * FS:
            j += 1
        if j < len(ft) and ft[j] <= ri + pat_range[1] * FS:
            R.append(ri), F.append(ft[j]), P.append(pk[j])
            j += 1
    return BeatPairs(np.asarray(R, int), np.asarray(F, int), np.asarray(P, int), ecg, ppg,
                     meta=dict(n_r=len(r), n_pulses=len(pk)))
