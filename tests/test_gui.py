"""The window, driven without a screen (Qt offscreen): load models, drop a file with PPG + ECG columns, run,
read the metrics, change display options, remove the ECG, drop a too-short file. Skipped if PySide6 is absent."""
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from conftest import write_csv  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from studio import config  # noqa: E402
from studio.ui import main_window  # noqa: E402


def wait(app, cond, timeout=60.0):
    t = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.01)
        if time.time() - t > timeout:
            raise AssertionError("timed out")


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    from studio.ui import theme
    a.setStyleSheet(theme.QSS)
    return a


def test_window_flow(app, synth, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(main_window, "popup", lambda parent, title, text, **k: shown.append(title) or None)
    w = main_window.MainWindow(config.Settings(tmp_path / "settings.json"))
    w.show()
    wait(app, lambda: len(w.models) == len(w.specs))
    assert w.models["template"].status == "ready"
    assert not w.models_panel.run_all.isEnabled()                      # nothing loaded yet

    t = np.arange(len(synth.ppg)) / synth.ppg_fs
    f = write_csv(tmp_path / "rec.csv", [t, synth.ppg, synth.ecg], ["time", "ppg", "ecg"])
    w.open_file(str(f), "ppg")
    wait(app, lambda: w.aligned is not None and w.aligned[1] is not None)
    assert w.models_panel.run_all.isEnabled()
    assert "ppg" in w.plots.rows and "ecg" in w.plots.rows and "template" in w.plots.rows

    w.run(w.models_panel.selected())
    wait(app, lambda: not w.busy and "template" in w.results)
    r = w.results["template"]
    assert r.vs_true is not None and r.vs_true["r_f1_dc"] > 0.9
    assert w.plots.b_metrics.isEnabled()
    w.toggle_metrics()
    assert w.drawer.isVisible() and w.drawer.t_cmp.rowCount() == len(w.results)
    assert w.drawer.t_err.rowCount() == len(w.results) and w.drawer.err_stack.currentIndex() == 0

    for a in (w.plots.a_align, w.plots.a_band, w.plots.a_rpeaks):     # display options redraw without error
        a.setChecked(False)
        app.processEvents()
        a.setChecked(True)
    w.models_panel.length.setValue(5)
    w.models_panel.start.setValue(12)

    w.remove_input("ecg")
    assert w.aligned[1] is None and not w.results and "ecg" not in w.plots.rows
    w.run(["template"])
    wait(app, lambda: not w.busy and "template" in w.results)
    w.update_drawer()
    assert w.drawer.err_stack.currentIndex() == 1                       # "No true ECG was given"

    np.save(tmp_path / "short_250Hz.npy", synth.ppg[:1000])
    w.open_file(str(tmp_path / "short_250Hz.npy"), "ppg")
    wait(app, lambda: "File too short" in shown)
    assert w.ppg is not None and w.ppg.name == "rec.csv"                 # the earlier file stays
    w.close()
