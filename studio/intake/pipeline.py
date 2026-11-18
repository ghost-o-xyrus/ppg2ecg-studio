"""The whole intake: file -> one continuous 128 Hz signal, with every notice raised on the way.

    prep = prepare("recording.csv", "ppg")            # may raise NeedRate / NeedColumn / TooLong / IntakeError
    prep = prepare("recording.csv", "ppg", rate=125)  # ... answered
    ppg, ecg, notices = align(prep_ppg, prep_ecg)     # common time base, trimmed to the overlap
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import config
from . import checks, fs_adjust, window_adjust
from .notices import IntakeError, NeedColumn, NeedRate, Notice, TOAST, TooLong
from .readers import read_any


@dataclass
class Prepared:
    kind: str                     # 'ppg' | 'ecg'
    path: str
    name: str
    x: np.ndarray                 # continuous signal at 128 Hz (PPG possibly flipped upright)
    fs_in: float                  # rate of the file
    fs_source: str
    column: str
    fmt: str
    t0: float | None              # first time stamp of the file (s), if it has a time column
    n_in: int                     # samples in the file (after joining windows)
    seconds: float
    peaks: np.ndarray             # PPG systolic peaks or ECG R peaks (at 128 Hz)
    bpm: float                    # main pulse rate from the spectrum
    inverted: bool = False        # ECG drawn upside-down-corrected
    flipped: bool = False         # PPG flipped upright
    cropped_from: float | None = None
    notices: list = field(default_factory=list)
    embedded: dict = field(default_factory=dict)   # other signals in the same file: name -> (array, fs, t0)
    windows_shape: tuple | None = None

    @property
    def fs(self):
        return config.WORK_FS

    @property
    def n_beats(self):
        return int(len(self.peaks))


def _size_check(path, limit_mb):
    try:
        mb = os.path.getsize(path) / 1e6
    except OSError as e:
        raise IntakeError("Cannot open file", str(e)) from None
    if limit_mb and mb > limit_mb:
        raise IntakeError("File too large", f"This file is {mb:.0f} MB; the limit is {limit_mb:.0f} MB per file. "
                                            "Large-file mode (for recordings up to 2 h) arrives in a later "
                                            "version; for now, cut the file to 2 minutes.")


def from_array(x, fs, kind, *, path="", name="", fs_source="", column="", fmt="", t0=None, limit_s=config.MAX_SECONDS,
               keep_first=False, embedded=None, notes=()):
    """Shared tail of the intake for a signal already in memory (2-D = rows of windows)."""
    notices = [Notice(TOAST, "File read", n[0].upper() + n[1:] + ".") for n in notes]
    x = np.asarray(x, float)
    wshape = None
    if x.ndim == 2:
        wshape = x.shape
        x, n = window_adjust.join_windows(x, fs)
        if n:
            notices.append(n)
    x, n = window_adjust.fill_gaps(x, fs, config.GAP_FILL_S)
    if n:
        notices.append(n)
    seconds = len(x) / fs
    cropped = None
    try:
        checks.check_length(seconds, kind, limit_s)
    except TooLong:
        if not keep_first:
            raise
        cropped = seconds
        x = x[: int(round(limit_s * fs))]
        seconds = len(x) / fs
        notices.append(Notice(TOAST, "First part used", f"The file is {cropped / 60:.1f} min long; its first "
                                                        f"{limit_s / 60:.0f} min are used."))
    bpm, n = fs_adjust.check_rate(x, fs, kind)
    if n:
        notices.append(n)
    x128 = fs_adjust.to_work_rate(x, fs)
    inverted = flipped = False
    if kind == "ppg":
        y, peaks, ns = checks.ppg_checks(x128)
        flipped = y is not x128 and bool(np.any(y != x128))
        x128 = y
    else:
        peaks, inverted, ns = checks.ecg_checks(x128)
    notices += ns
    return Prepared(kind, str(path), name or Path(path).name, x128, float(fs), fs_source, column, fmt, t0, len(x),
                    len(x128) / config.WORK_FS, np.asarray(peaks, int), bpm, inverted, flipped, cropped, notices,
                    embedded or {}, wshape)


def prepare(path, kind="ppg", settings=None, rate=None, column=None, limit_s=config.MAX_SECONDS,
            limit_mb=config.MAX_MB, keep_first=False):
    _size_check(path, limit_mb)
    try:
        raw = read_any(path, kind, column)
    except (IntakeError, NeedColumn, NeedRate):
        raise
    except Exception as e:          # malformed file: report, never crash
        raise IntakeError("Cannot read file", f"{Path(path).name}: {e}") from None
    fs, src = fs_adjust.find_rate(raw, path, settings, rate)
    if not (1.0 <= fs <= 100000.0):
        raise IntakeError("Sampling rate not usable", f"{fs:g} Hz is not a usable sampling rate.")
    emb = {k: (v, fs, raw.t0) for k, v in raw.other_columns.items()}
    return from_array(raw.data, fs, kind, path=path, fs_source=src, column=raw.column, fmt=raw.fmt, t0=raw.t0,
                      limit_s=limit_s, keep_first=keep_first, embedded=emb, notes=raw.notes)


def prepare_embedded(prep: Prepared, name, kind="ecg", limit_s=config.MAX_SECONDS, keep_first=False):
    """A second signal found in the same file (e.g. lead II next to PLETH)."""
    x, fs, t0 = prep.embedded[name]
    return from_array(x, fs, kind, path=prep.path, name=f"{prep.name} [{name}]", fs_source=prep.fs_source,
                      column=name, fmt=prep.fmt, t0=t0, limit_s=limit_s, keep_first=keep_first)


def align(ppg: Prepared, ecg: Prepared | None):
    """Put PPG and true ECG on one time base (time stamps when both files have them, else both start at 0)
    and trim to the overlap. -> (ppg128, ecg128 or None, notices)."""
    if ecg is None:
        return ppg.x, None, []
    fs = config.WORK_FS
    notices = []
    a, b = ppg.x, ecg.x
    if ppg.t0 is not None and ecg.t0 is not None and abs(ppg.t0 - ecg.t0) > 1.0 / fs:
        d = int(round((ecg.t0 - ppg.t0) * fs))       # ECG starts d samples after the PPG
        if d > 0:
            a = a[d:]
        else:
            b = b[-d:]
        notices.append(Notice(TOAST, "Aligned by time stamps", f"The true ECG starts {d / fs:+.2f} s from the PPG; "
                                                               "both were aligned by their time stamps."))
    n = min(len(a), len(b))
    if abs(len(a) - len(b)) > fs:
        notices.append(Notice(TOAST, "Different lengths", f"PPG {len(a) / fs:.0f} s and true ECG {len(b) / fs:.0f} s "
                                                          f"differ; the common {n / fs:.0f} s are compared."))
    if n / fs < config.MIN_SECONDS:
        raise IntakeError("Too little overlap", f"The PPG and true ECG overlap for only {n / fs:.1f} s; at least "
                                                f"{config.MIN_SECONDS:.0f} s are needed.")
    return a[:n], b[:n], notices


def aligned_peaks(ppg: Prepared, x_aligned):
    """PPG pulses on the aligned (possibly trimmed) PPG; none if the intake found no pulse wave."""
    from bench import dsp
    if ppg.n_beats == 0:
        return np.array([], int)
    return dsp.detect_ppg_peaks(np.asarray(x_aligned, float), config.WORK_FS)
