"""Start the window: python -m studio  (run.bat does this with the project's own Python)."""
from __future__ import annotations

import os
import sys
import threading
import traceback


def _fatal(text):
    """Last-resort message box when Qt itself cannot start (missing package, broken install)."""
    print(text, file=sys.stderr)
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, text, "PPG2ECG Studio", 0x10)
        except Exception:
            pass


def _warm_up():
    """Load the signal-processing code while the user looks at the window, so the first file opens quickly."""
    try:
        import bench.metrics  # noqa: F401
        import bench.runner  # noqa: F401
        from .intake import pipeline  # noqa: F401
    except Exception:
        traceback.print_exc()


def main(argv=None):
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFont, QGuiApplication
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        _fatal(f"PySide6 is not installed ({e}). Run setup.bat first.")
        return 1
    if sys.platform == "win32":
        try:   # own taskbar icon instead of Python's
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("PPG2ECG.Studio")
        except Exception:
            pass
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("PPG2ECG Studio")
    app.setStyle("Fusion")
    from . import config
    from .ui import theme
    from .ui.main_window import MainWindow, app_icon
    app.setFont(QFont(theme.FONT, 10))
    app.setStyleSheet(theme.QSS)
    app.setWindowIcon(app_icon())
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True, background="w", foreground=theme.MUTED)

    def excepthook(t, v, tb):          # an unexpected error is shown, not silently fatal
        text = "".join(traceback.format_exception(t, v, tb))
        print(text, file=sys.stderr)
        try:
            from .ui.widgets import popup
            popup(None, "Unexpected error", text[-1500:], big=False, dim=False)
        except Exception:
            pass
    sys.excepthook = excepthook

    w = MainWindow(config.Settings())
    w.show()
    threading.Thread(target=_warm_up, daemon=True).start()
    for a in (argv or sys.argv)[1:]:        # files given on the command line / dropped on the shortcut
        if os.path.isfile(a):
            w.open_file(a, "ppg")
            break
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
