"""App-wide constants, paths and the small per-user settings file."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

APP_NAME = "PPG2ECG Studio"
APP_VERSION = "0.2.0"

ROOT = Path(__file__).resolve().parents[1]          # repository folder
MODELS_DIR = ROOT / "models"
WEIGHTS_DIR = Path(os.environ.get("PPG2ECG_STUDIO_WEIGHTS", ROOT / "weights"))   # override for tests

def _release_url() -> str:
    """Where `setup.py` downloads OUR weights from: the GitHub Release of this version, of the repository this
    copy was cloned from (read from .git/config, so no account name is written into the code). Override with
    the PPG2ECG_RELEASE environment variable, or install from files with `setup.py --offline FOLDER`."""
    import re
    try:
        cfg = (ROOT / ".git" / "config").read_text(encoding="utf-8", errors="ignore")
        m = re.search(r'\[remote "origin"\][^\[]*?url\s*=\s*\S*github\.com[:/]([^/\s]+)/([^/\s]+?)(?:\.git)?\s',
                      cfg + "\n")
        if m:
            return f"https://github.com/{m.group(1)}/{m.group(2)}/releases/download/v{APP_VERSION}"
    except OSError:
        pass
    return f"https://github.com/YOUR_GITHUB_USERNAME/ppg2ecg-studio/releases/download/v{APP_VERSION}"


RELEASE_URL = os.environ.get("PPG2ECG_RELEASE") or _release_url()

# setup.py flag of each built-in method (setup.bat --chiu ...)
SETUP_FLAGS = {"zhu_dct": "zhu", "cardiogan": "cardiogan", "ppgflowecg": "ppgflowecg", "rddm": "rddm",
               "clep_gan": "clepgan", "chiu2020": "chiu", "p2e_wgan": "p2e", "yalcin2026": "yalcin"}

# ---------------------------------------------------------------- signal rules
WORK_FS = 128.0                 # every input is converted to this rate once
MIN_SECONDS = 10.0              # longest model window (template, Zhu DCT, PPGFlowECG)
MAX_SECONDS = 120.0             # default-size limit (large-file mode arrives in v0.4)
MAX_MB = 20.0
FEW_BEATS = 20                  # fewer detected beats than this -> side notification
FEW_SECONDS = 30.0
HR_RANGE_BPM = (36.0, 210.0)    # plausible pulse rate; outside it the sampling rate looks wrong
GAP_FILL_S = 1.0                # NaN gaps up to this long are interpolated
COMMON_RATES = [25, 32, 50, 64, 100, 125, 128, 250, 256, 300, 360, 500, 1000]

PLOT_MIN_S, PLOT_MAX_S, PLOT_DEFAULT_S = 2, 15, 10
SPINNER_DELAY_MS = 300          # show "Adjusting data to preferred format..." after this long

# scorer settings: identical to Track 1
EVAL_WINDOW_S = 10.0
FD_MAX_WINDOWS = 20
STITCH_OVERLAP = 0.5


def user_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    d = base / APP_NAME
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return d


class Settings:
    """Tiny JSON settings store: remembered sampling rates per folder, device override, display options."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else user_dir() / "settings.json"
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.save()

    def save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        except OSError:
            pass

    # rate remembered for files from the same folder
    def folder_rate(self, folder: str) -> float | None:
        v = self.data.get("folder_rates", {}).get(os.path.normcase(str(folder)))
        return float(v) if v else None

    def remember_folder_rate(self, folder: str, fs: float):
        self.data.setdefault("folder_rates", {})[os.path.normcase(str(folder))] = float(fs)
        self.save()
