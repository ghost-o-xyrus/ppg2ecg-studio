"""Messages raised while reading and converting a file, and the questions intake may need answered."""
from __future__ import annotations

from dataclasses import dataclass

# levels, from most to least intrusive
MODAL = "modal"     # big central pop-up that dims the window (blocks or misleads: file too short ...)
POPUP = "popup"     # dismissable notice (no heartbeats found ...)
TOAST = "toast"     # side notification, bottom right, disappears after 8 s


@dataclass
class Notice:
    level: str
    title: str
    text: str

    def __str__(self):
        return f"[{self.level}] {self.title}: {self.text}"


class IntakeError(Exception):
    """The file cannot be used. Shown as a central pop-up; Run stays disabled."""

    def __init__(self, title: str, text: str):
        super().__init__(f"{title}: {text}")
        self.title, self.text = title, text


class NeedRate(Exception):
    """The sampling rate could not be found in the file; ask the user once (remembered per folder)."""

    def __init__(self, path: str, guess: float | None = None):
        super().__init__(f"sampling rate needed for {path}")
        self.path, self.guess = path, guess


class NeedColumn(Exception):
    """Several columns could hold the signal; ask which one."""

    def __init__(self, path: str, candidates: list[str]):
        super().__init__(f"column needed for {path}: {candidates}")
        self.path, self.candidates = path, candidates


class TooLong(Exception):
    """Longer than the default limit; the user may keep the first part (large-file mode comes later)."""

    def __init__(self, seconds: float, limit: float):
        super().__init__(f"{seconds:.0f} s > {limit:.0f} s")
        self.seconds, self.limit = seconds, limit
