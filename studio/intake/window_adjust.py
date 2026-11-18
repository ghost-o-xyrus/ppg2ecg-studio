"""Window step: turn whatever layout the file has into one continuous signal.

* 2-D arrays are rows of windows (4 s, 16 s, anything). Rows are joined back in order, assuming they are
  consecutive. If consecutive rows repeat each other (sliding windows with overlap), the overlap is found
  exactly and kept once, so a 50%-overlap window array is not doubled in length.
* Missing samples (NaN / inf): gaps up to 1 s are interpolated; at longer gaps the signal is split and the
  longest piece is kept (reported).

Each model later cuts its own windows from the continuous 128 Hz signal (bench.runner), as in Track 1.
"""
from __future__ import annotations

import numpy as np

from .notices import Notice, TOAST


def find_overlap(W, rtol=1e-6) -> int:
    """Number of samples by which consecutive rows repeat each other (0 = no overlap)."""
    n, L = W.shape
    if n < 2:
        return 0
    a, b = W[0], W[1]
    scale = max(np.nanmax(np.abs(W[:2])), 1e-12)
    for k in range(L - 1, 0, -1):            # largest overlap first (most specific)
        if np.allclose(a[L - k:], b[:k], rtol=0, atol=rtol * scale, equal_nan=True):
            ok = all(np.allclose(W[i][L - k:], W[i + 1][:k], rtol=0, atol=rtol * scale, equal_nan=True)
                     for i in range(1, min(n - 1, 8)))
            if ok and k >= 2:
                return k
    return 0


def join_windows(W, fs=None):
    """(n, L) rows -> (1-D signal, notice or None)."""
    W = np.asarray(W, float)
    n, L = W.shape
    if n == 1:
        return W[0], None
    k = find_overlap(W)
    x = np.concatenate([W[0]] + [w[k:] for w in W[1:]])
    if fs:
        lw = f"{L / fs:.3g} s"
        total = f"{len(x) / fs:.0f} s"
    else:
        lw, total = f"{L} samples", f"{len(x)} samples"
    extra = f", overlapping by {k} samples (kept once)" if k else ""
    return x, Notice(TOAST, "Windows joined", f"{n} windows of {lw}{extra} were joined into one {total} signal, "
                                               "assuming they are consecutive.")


def fill_gaps(x, fs, max_gap_s=1.0):
    """-> (signal, notice or None). Short gaps interpolated; at a longer gap the longest finite piece is kept."""
    x = np.asarray(x, float).copy()
    bad = ~np.isfinite(x)
    if not bad.any():
        return x, None
    if bad.all():
        return x[:0], None
    edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(np.int8), 0]))
    runs = list(zip(edges[::2], edges[1::2]))
    n_filled = 0
    for a, b in runs:
        if (b - a) / fs <= max_gap_s and a > 0 and b < len(x):
            x[a:b] = np.interp(np.arange(a, b), [a - 1, b], [x[a - 1], x[b]])
            n_filled += 1
    bad = ~np.isfinite(x)
    if not bad.any():
        return x, Notice(TOAST, "Gaps filled", f"{n_filled} short gap{'s' if n_filled > 1 else ''} (up to "
                                                f"{max_gap_s:g} s) in the signal were filled by interpolation.")
    good = np.r_[0, (~bad).astype(np.int8), 0]
    e = np.flatnonzero(np.diff(good))
    pieces = list(zip(e[::2], e[1::2]))
    a, b = max(pieces, key=lambda ab: ab[1] - ab[0])
    return x[a:b], Notice(TOAST, "Signal has gaps",
                          f"The signal has gaps longer than {max_gap_s:g} s; the longest continuous part "
                          f"({(b - a) / fs:.0f} s from {a / fs:.0f} s) is used.")
