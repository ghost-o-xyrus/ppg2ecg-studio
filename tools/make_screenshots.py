"""Render the README screenshots off-screen from a synthetic recording (bench.datasets.make_synthetic).

    .venv\\Scripts\\python tools\\make_screenshots.py [--out docs\\screenshots]

Writes start.png (empty window), main.png (after Run all, with a true ECG) and metrics.png (metrics drawer,
Errors vs true ECG). No real recording is used: the PPG and ECG are generated here, written to a temporary
CSV and loaded through the app's normal intake.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32" and "QT_QPA_FONTDIR" not in os.environ:    # the off-screen platform needs real fonts
    os.environ["QT_QPA_FONTDIR"] = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")

import numpy as np  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs" / "screenshots"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    import pyqtgraph as pg

    from bench.datasets import make_synthetic
    from studio import config
    from studio.ui import theme
    from studio.ui.main_window import MainWindow
    app.setFont(QFont(theme.FONT, 10))
    app.setStyleSheet(theme.QSS)
    pg.setConfigOptions(antialias=True, background="w", foreground=theme.MUTED)

    def pump(seconds=0.3):
        t = time.time()
        while time.time() - t < seconds:
            app.processEvents()
            time.sleep(0.01)

    def wait(cond, timeout=120):
        t = time.time()
        while not cond():
            app.processEvents()
            time.sleep(0.01)
            if time.time() - t > timeout:
                raise SystemExit("timed out waiting for the window")

    def shot(name):
        for t in list(w.toasts.toasts):          # side notifications would cover the plots
            t.close_now()
        pump(0.4)
        path = out / name
        w.grab().save(str(path))
        print(f"wrote {path}")

    with tempfile.TemporaryDirectory() as tmp:
        w = MainWindow(config.Settings(Path(tmp) / "settings.json"))
        w.resize(1420, 900)
        w.show()
        wait(lambda: len(w.models) == len(w.specs))
        shot("start.png")

        r = make_synthetic(subject="demo", duration=90.0, fs=250.0, hr=68.0, seed=7)
        t = np.arange(len(r.ppg)) / r.ppg_fs
        f = Path(tmp) / "synthetic_recording.csv"
        np.savetxt(f, np.column_stack([t, r.ppg, r.ecg]), delimiter=",", fmt="%.6f",
                   header="time,ppg,ecg", comments="")
        w.open_file(str(f), "ppg")
        wait(lambda: w.aligned is not None and w.aligned[1] is not None)
        w.run(w.models_panel.selected())
        wait(lambda: not w.busy and w.results)
        w.drawer.close_drawer()
        shot("main.png")

        if not w.drawer.isVisible():
            w.toggle_metrics()
        w.drawer.tabs.setCurrentIndex(1)
        shot("metrics.png")
        w.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
