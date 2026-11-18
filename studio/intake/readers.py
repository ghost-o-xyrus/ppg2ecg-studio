"""Read CSV / TXT / NPY / NPZ files into a raw signal plus whatever the file says about its sampling rate.

Nothing here resamples or judges the signal; that is fs_adjust.py, window_adjust.py and checks.py.

CSV / TXT  one numeric column (the signal), or time + signal, or several columns with a header. The
           delimiter and header are detected; non-numeric rows are dropped; '#' lines are comments and
           '# fs=125' (or 'fs: 125', 'sampling rate = 125 Hz') gives the rate.
NPY        1-D array = one signal; 2-D array = rows of windows ((L, 1) and (1, L) are one signal;
           (n, 2) with an increasing first column = time + signal).
NPZ        arrays named ppg / pleth / bvp / ecg / ii / signal / x / data, optional fs.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .notices import IntakeError, NeedColumn

PPG_NAMES = ("ppg", "pleth", "bvp", "pulse", "spo2wave", "pg")
ECG_NAMES = ("ecg", "ii", "mlii", "lead ii", "lead2", "ekg", "lii")
GENERIC_NAMES = ("signal", "x", "data", "value", "y")
FS_NAMES = ("fs", "fs_hz", "sampling_rate", "samplingrate", "sample_rate", "rate", "srate", "freq")
EXTS = (".csv", ".txt", ".tsv", ".dat", ".npy", ".npz")

_FS_COMMENT = re.compile(r"(?:\bfs\b|sampling[ _-]?(?:rate|frequency)|sample[ _-]?rate|srate)\s*[=:]?\s*"
                         r"([0-9]+(?:\.[0-9]+)?)", re.I)
_FS_NAME = re.compile(r"(?<![0-9.])([0-9]+(?:\.[0-9]+)?)\s*hz(?![a-z])", re.I)


@dataclass
class RawSignal:
    data: np.ndarray                    # 1-D signal or 2-D (rows of windows)
    fs: float | None = None             # sampling rate if the file states it
    fs_source: str = ""                 # 'time column', 'comment', 'file name', 'npz fs', ...
    column: str = ""                    # column / array the signal came from
    t0: float | None = None             # first time stamp (s), when there is a time column
    fmt: str = ""                       # csv | txt | npy | npz
    other_columns: dict = field(default_factory=dict)   # name -> 1-D array (e.g. an ECG in the PPG file)
    notes: list = field(default_factory=list)


def fs_from_name(path) -> float | None:
    m = _FS_NAME.search(Path(path).stem.replace("_", " "))
    return float(m.group(1)) if m else None


def _norm(name) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def _tokens(name) -> set[str]:
    return set(t for t in re.split(r"[^a-z0-9]+", _norm(name)) if t)


def _match(name, names) -> bool:
    n, tok = _norm(name), _tokens(name)
    for k in names:
        if " " in k:
            if k in n:
                return True
        elif k in tok or (len(k) >= 3 and n.startswith(k)):
            return True
    return False


def _is_time_name(name) -> bool:
    tok = _tokens(name)
    return bool(tok & {"time", "t", "sec", "secs", "seconds", "s", "ms", "timestamp", "timestamps", "elapsed"}) \
        and not (tok & {"ppg", "ecg", "pleth", "ii", "bvp"})


def _time_unit_scale(name, step) -> float:
    """Seconds per unit of a time column: ms if the name says so or the step is implausibly long for seconds."""
    tok = _tokens(name)
    if tok & {"ms", "msec", "millis", "milliseconds"}:
        return 1e-3
    if tok & {"us", "usec", "micros"}:
        return 1e-6
    return 1e-3 if step >= 1.0 else 1.0      # a PPG sampled slower than 1 Hz is not a PPG


def _looks_like_time(x) -> bool:
    """An (almost) evenly spaced, non-decreasing column. Rounded time stamps are allowed (BIDMC prints
    2 decimals after 100 s, so its steps alternate 0 and 0.01 s at 125 Hz)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10 or x[-1] <= x[0]:
        return False
    d = np.diff(x)
    if np.mean(d >= 0) < 0.999:
        return False
    step = (x[-1] - x[0]) / (len(x) - 1)
    resid = np.abs(x - (x[0] + step * np.arange(len(x))))
    return bool(np.max(resid) < max(5 * step, 0.01 * (x[-1] - x[0])))


def time_step(t) -> float:
    """Mean sampling step from the span, robust to rounded time stamps."""
    t = np.asarray(t, float)
    t = t[np.isfinite(t)]
    return float((t[-1] - t[0]) / (len(t) - 1))


def pick(columns, kind, path="") -> str | None:
    """Choose the signal column for kind ('ppg' or 'ecg'). Returns None when it should be asked."""
    names = PPG_NAMES if kind == "ppg" else ECG_NAMES
    hits = [c for c in columns if _match(c, names)]
    if kind == "ecg" and len(hits) > 1:     # prefer lead II when several leads are present
        ii = [c for c in hits if _tokens(c) & {"ii", "mlii", "lii"}]
        hits = ii or hits
    if hits:
        return hits[0]
    gen = [c for c in columns if _match(c, GENERIC_NAMES)]
    if len(gen) == 1:
        return gen[0]
    if len(columns) == 1:
        return columns[0]
    return None


# ---------------------------------------------------------------- text files
def _sniff_delimiter(lines) -> str | None:
    sample = "\n".join(lines[:50])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        pass
    for d in (",", ";", "\t", "|"):
        if all(d in ln for ln in lines[:20] if ln.strip()):
            return d
    return None                              # whitespace-separated (or a single column)


def _is_number(s) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_text(path, kind, column=None) -> RawSignal:
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    fs, fs_src = None, ""
    body = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith(("#", "%", "//")):
            m = _FS_COMMENT.search(s)
            if m and fs is None:
                fs, fs_src = float(m.group(1)), "comment line"
            continue
        body.append(ln)
    if not body:
        raise IntakeError("Empty file", f"{Path(path).name} has no numbers in it.")
    delim = _sniff_delimiter(body)
    split = (lambda s: s.split(delim)) if delim else (lambda s: s.split())
    first = [c.strip().strip('"').strip("'") for c in split(body[0])]
    has_header = not all(_is_number(c) for c in first if c != "")
    sep = delim if delim else r"\s+"
    df = pd.read_csv(io.StringIO("\n".join(body)), sep=sep, header=0 if has_header else None,
                     engine="python", skipinitialspace=True, dtype=str)
    if not has_header:
        df.columns = [f"column {i + 1}" for i in range(df.shape[1])]
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    num = df.apply(lambda s: pd.to_numeric(s.str.strip().str.strip('"'), errors="coerce"))
    num = num.loc[:, num.notna().mean() > 0.5]              # drop text columns (labels, dates ...)
    dropped = int(num.isna().all(axis=1).sum())
    num = num[~num.isna().all(axis=1)]
    if num.shape[1] == 0 or len(num) == 0:
        raise IntakeError("No numbers found", f"{Path(path).name} has no numeric column.")
    out = RawSignal(np.array([]), fs, fs_src, fmt=Path(path).suffix.lower().lstrip("."))
    if dropped:
        out.notes.append(f"{dropped} non-numeric row{'s' if dropped > 1 else ''} skipped")

    cols = list(num.columns)
    tcol = next((c for c in cols if has_header and _is_time_name(c) and _looks_like_time(num[c])), None)
    if tcol is None and len(cols) >= 2 and _looks_like_time(num[cols[0]]) and not has_header:
        tcol = cols[0]
    sig_cols = [c for c in cols if c != tcol]
    if column is not None:
        if column not in sig_cols:
            raise IntakeError("Column not found", f"{Path(path).name} has no column '{column}'.")
        col = column
    else:
        col = pick(sig_cols, kind, path) if has_header else (sig_cols[0] if len(sig_cols) == 1 else None)
        if col is None:
            raise NeedColumn(str(path), sig_cols)
    out.data = num[col].to_numpy(float)
    out.column = col
    if tcol is not None:
        t = num[tcol].to_numpy(float)
        step = time_step(t)
        scale = _time_unit_scale(tcol, step)
        if step > 0:
            out.fs, out.fs_source = 1.0 / (step * scale), f"time column '{tcol}'"
            out.t0 = float(t[np.isfinite(t)][0] * scale)
    other = "ecg" if kind == "ppg" else "ppg"
    oc = pick([c for c in sig_cols if c != col], other) if has_header else None
    if oc is not None and _match(oc, ECG_NAMES if other == "ecg" else PPG_NAMES):
        out.other_columns[oc] = num[oc].to_numpy(float)
    return out


# ---------------------------------------------------------------- numpy
def _fs_from_npz(z) -> float | None:
    for k in z.files:
        if _norm(k) in FS_NAMES:
            try:
                return float(np.asarray(z[k]).ravel()[0])
            except (ValueError, IndexError):
                pass
    return None


def _shape_signal(a, name):
    a = np.asarray(a)
    if a.dtype.kind not in "fiu":
        raise IntakeError("Not a signal", f"array '{name}' is not numeric ({a.dtype}).")
    a = np.squeeze(a.astype(float))
    if a.ndim == 0:
        raise IntakeError("Not a signal", f"array '{name}' holds a single number.")
    if a.ndim > 2:
        raise IntakeError("Unsupported shape", f"array '{name}' has shape {a.shape}; use 1-D (one signal) or "
                                                "2-D (rows of windows).")
    return a


def _split_time(a):
    """(n, 2) arrays whose first column is an increasing time axis -> (time, signal)."""
    if a.ndim == 2 and a.shape[1] == 2 and a.shape[0] > 20 and _looks_like_time(a[:, 0]):
        return a[:, 0], a[:, 1]
    if a.ndim == 2 and a.shape[0] == 2 and a.shape[1] > 20 and _looks_like_time(a[0]):
        return a[0], a[1]
    return None, a


def read_npy(path, kind, column=None) -> RawSignal:
    try:
        a = np.load(path, allow_pickle=False)
    except ValueError as e:
        raise IntakeError("Cannot read file", f"{Path(path).name}: {e}") from None
    a = _shape_signal(a, Path(path).name)
    out = RawSignal(np.array([]), fmt="npy", column="array")
    t, a = _split_time(a)
    out.data = a
    if t is not None:
        step = time_step(t)
        scale = _time_unit_scale("", step)
        out.fs, out.fs_source, out.t0 = 1.0 / (step * scale), "time column", float(t[0] * scale)
    return out


def read_npz(path, kind, column=None) -> RawSignal:
    with np.load(path, allow_pickle=False) as z:
        keys = [k for k in z.files if _norm(k) not in FS_NAMES and np.asarray(z[k]).size > 1]
        if column is not None:
            key = column
        else:
            key = pick(keys, kind, path)
            if key is None:
                raise NeedColumn(str(path), keys)
        if key not in z.files:
            raise IntakeError("Array not found", f"{Path(path).name} has no array '{key}'.")
        out = RawSignal(_shape_signal(z[key], key), fmt="npz", column=key)
        fs = _fs_from_npz(z)
        if fs:
            out.fs, out.fs_source = fs, "fs array"
        other = "ecg" if kind == "ppg" else "ppg"
        ok = pick([k for k in keys if k != key], other)
        if ok is not None and _match(ok, ECG_NAMES if other == "ecg" else PPG_NAMES):
            arr = np.squeeze(np.asarray(z[ok], float))
            if arr.ndim in (1, 2):
                out.other_columns[ok] = arr
    t, out.data = _split_time(out.data)
    if t is not None and out.fs is None:
        step = time_step(t)
        sc = _time_unit_scale("", step)
        out.fs, out.fs_source, out.t0 = 1.0 / (step * sc), "time column", float(t[0] * sc)
    return out


def read_any(path, kind="ppg", column=None) -> RawSignal:
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in EXTS:
        raise IntakeError("File type not supported", f"{p.name}: use CSV, TXT, NPY or NPZ.")
    if ext == ".npy":
        out = read_npy(p, kind, column)
    elif ext == ".npz":
        out = read_npz(p, kind, column)
    else:
        out = read_text(p, kind, column)
    if out.fs is None:
        fs = fs_from_name(p)
        if fs:
            out.fs, out.fs_source = fs, "file name"
    return out
