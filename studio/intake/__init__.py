"""Data intake: any PPG / ECG file -> one continuous 128 Hz signal (readers, fs_adjust, window_adjust, checks)."""
from .notices import IntakeError, MODAL, NeedColumn, NeedRate, Notice, POPUP, TOAST, TooLong  # noqa: F401
from .pipeline import Prepared, align, aligned_peaks, from_array, prepare, prepare_embedded  # noqa: F401
