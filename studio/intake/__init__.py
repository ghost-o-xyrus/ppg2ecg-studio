"""Data intake: any PPG / ECG file -> one continuous 128 Hz signal (readers, fs_adjust, window_adjust, checks)."""
from .notices import IntakeError, MODAL, NeedColumn, NeedRate, Notice, POPUP, TOAST, TooLong  # noqa: F401

_PIPELINE = ("Prepared", "align", "aligned_peaks", "from_array", "prepare", "prepare_embedded")


def __getattr__(name):
    """The pipeline (pandas, scipy) is imported on first use, so the window can open before it is loaded."""
    if name in _PIPELINE:
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["IntakeError", "MODAL", "NeedColumn", "NeedRate", "Notice", "POPUP", "TOAST", "TooLong", *_PIPELINE]
