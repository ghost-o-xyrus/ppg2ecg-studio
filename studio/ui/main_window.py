"""The main window: input (top left), models (top right), plots (bottom), metrics drawer, notifications."""
from __future__ import annotations

import os
import time

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QIcon, QKeySequence, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QFileDialog, QInputDialog, QLabel, QMainWindow, QSplitter, QVBoxLayout, QWidget

from .. import config, engine, intake
from ..engine import device as devmod
from ..engine.process import Engine, EngineCrashed
from ..intake import IntakeError, NeedColumn, NeedRate, TooLong
from ..scoring import display_shift
from . import theme
from .input_panel import InputPanel
from .metrics_drawer import MetricsDrawer
from .model_panel import ModelPanel
from .plots import PlotArea, band, scale, shift
from .widgets import FILE_FILTER, RateDialog, ToastHost, dispatcher, fmt_duration, in_gui, popup, start


def app_icon() -> QIcon:
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(theme.BLUE))
    p.drawRoundedRect(2, 2, 60, 60, 14, 14)
    path = QPainterPath(QPointF(8, 36))
    for x, y in ((20, 36), (25, 30), (29, 36), (33, 36), (37, 12), (41, 50), (45, 36), (56, 36)):
        path.lineTo(QPointF(x, y))
    pen = QPen(QColor("white"), 4.5)
    pen.setJoinStyle(Qt.RoundJoin)
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawPath(path)
    p.end()
    return QIcon(pm)


class MainWindow(QMainWindow):
    def __init__(self, settings: config.Settings):
        super().__init__()
        self.settings = settings
        self.setWindowTitle(config.APP_NAME)
        self.setWindowIcon(app_icon())
        self.resize(1420, 900)
        self.setMinimumSize(QSize(1080, 700))
        self.setAcceptDrops(True)

        self.device = devmod.probe(settings.get("device", "auto"))
        self.specs = engine.discover()
        self.models: dict = {}           # id -> engine.process.ModelState once loaded (or failed)
        self.engine = None
        self.model_src = None            # the aligned PPG at the file's own rate (what the models get)
        self.ppg = None                 # intake.Prepared
        self.ecg = None
        self.ecg_from_ppg_file = False
        self.aligned = None             # (ppg128, ecg128 or None, ppg peaks, true R peaks)
        self.entry = 0                  # bumps whenever the inputs change; stale results are dropped
        self.results: dict[str, engine.Result] = {}
        self.busy = False
        self.pending: set = set()       # running intake jobs (spinner)
        self.told_no_ecg = -1

        self._build()
        self._menus()
        QTimer.singleShot(0, self.load_models)

    # ================================================================ layout
    def _build(self):
        central = QWidget()
        central.setObjectName("central")
        outer = QVBoxLayout(central)
        outer.setContentsMargins(12, 10, 12, 8)
        self.input = InputPanel()
        self.models_panel = ModelPanel()
        self.plots = PlotArea()
        top = QSplitter(Qt.Horizontal)
        top.addWidget(self.input)
        top.addWidget(self.models_panel)
        top.setSizes([1, 1])
        top.setChildrenCollapsible(False)
        top.setHandleWidth(10)
        main = QSplitter(Qt.Vertical)
        main.addWidget(top)
        main.addWidget(self.plots)
        main.setSizes([360, 520])
        main.setChildrenCollapsible(False)
        main.setHandleWidth(10)
        outer.addWidget(main)
        self.setCentralWidget(central)
        self.drawer = MetricsDrawer(self.plots.stage)
        self.toasts = ToastHost(central)

        self.input.fileChosen.connect(lambda f, k: self.open_file(f, k))
        self.input.removed.connect(self.remove_input)
        self.models_panel.set_models(self.specs)
        self.models_panel.runAll.connect(lambda: self.run(self.models_panel.selected()))
        self.models_panel.runOne.connect(lambda i: self.run([i]))
        self.models_panel.windowChanged.connect(self.plots.set_window)
        self.models_panel.set_input(None)
        self.plots.showMetrics.connect(self.toggle_metrics)
        self.plots.optionsChanged.connect(self.redraw)
        self.plots.hover.connect(lambda s: self.s_hover.setText(s))

        sb = self.statusBar()
        self.s_device = QLabel(f"CPU: {devmod.cpu_name()}")
        self.s_engine = QLabel("Models: loading…")
        self.s_hover = QLabel("")
        sb.addWidget(self.s_device)
        sb.addWidget(self.s_engine)
        sb.addPermanentWidget(self.s_hover)

    def _menus(self):
        m = self.menuBar()
        f = m.addMenu("&File")
        a = QAction("Open PPG…", self)
        a.setShortcut(QKeySequence(QKeySequence.StandardKey.Open))
        a.triggered.connect(lambda: self._browse("ppg"))
        f.addAction(a)
        a = QAction("Open true ECG…", self, shortcut=QKeySequence("Ctrl+E"))
        a.triggered.connect(lambda: self._browse("ecg"))
        f.addAction(a)
        a = QAction("Run all", self, shortcut=QKeySequence("Ctrl+R"))
        a.triggered.connect(lambda: self.models_panel.run_all.isEnabled() and self.run(self.models_panel.selected()))
        f.addAction(a)
        f.addSeparator()
        e = QAction("Export… (arrives in a later version)", self)
        e.setEnabled(False)
        f.addAction(e)
        f.addSeparator()
        a = QAction("Exit", self, shortcut=QKeySequence("Ctrl+Q"))
        a.triggered.connect(self.close)
        f.addAction(a)
        s = m.addMenu("&Settings")
        a = QAction(f"Device: automatic ({devmod.describe(self.device)})", self)
        a.setEnabled(False)
        s.addAction(a)
        a = QAction("Keep models ready after closing (arrives in a later version)", self)
        a.setEnabled(False)
        s.addAction(a)
        h = m.addMenu("&Help")
        a = QAction("Read me", self)
        a.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(config.ROOT / "README.md"))))
        h.addAction(a)
        a = QAction(f"About {config.APP_NAME}", self)
        a.triggered.connect(self.about)
        h.addAction(a)

    def about(self):
        popup(self, f"{config.APP_NAME} {config.APP_VERSION}",
              "Reconstructs an ECG from a PPG with every installed method, plots them against the PPG and "
              "(optionally) the true ECG, and measures them with the Track 1 scorer of the PPG2ECG thesis.\n\n"
              "This version: all nine methods of the thesis; export, cache and large-file mode follow in later "
              "versions.", big=False, dim=False)

    def _browse(self, kind):
        what = "PPG" if kind == "ppg" else "true ECG"
        path, _ = QFileDialog.getOpenFileName(self, f"Open {what} file", "", FILE_FILTER)
        if path:
            self.input.tabs.setCurrentIndex(0 if kind == "ppg" else 1)
            self.open_file(path, kind)

    # drop anywhere in the window -> the open input tab
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            if u.isLocalFile() and os.path.isfile(u.toLocalFile()):
                kind = "ppg" if self.input.tabs.currentIndex() == 0 else "ecg"
                self.open_file(u.toLocalFile(), kind)
                e.acceptProposedAction()
                return

    # ================================================================ models
    def load_models(self):
        """Models load in the engine process (window stays responsive); each reports when it is ready."""
        dispatcher()                     # created here, in the GUI thread: the engine reports from another
        self.engine = Engine(on_status=lambda st: in_gui(self._model_loaded, st),
                             on_restart=lambda: in_gui(self._engine_restarted))
        self.engine.load(self.specs, self.device)

    def _engine_restarted(self, _=None):
        self.models = {}
        self.toasts.show("Engine restarted", "A model stopped the engine. It was started again and the models are "
                                             "loading; press Run all once they are ready.")

    def closeEvent(self, e):
        if self.engine is not None:
            self.engine.close()
        super().closeEvent(e)

    def _model_loaded(self, lm):
        if lm.status == engine.LOADING:
            self.models.pop(lm.spec.id, None)
            self.models_panel.set_status(lm.spec.id, lm.status)
            return
        self.models[lm.spec.id] = lm
        self.models_panel.set_status(lm.spec.id, lm.status, lm.error)
        if lm.status == engine.FAILED:
            self.toasts.show(f"{lm.spec.name} could not load", lm.error[:200])
        if lm.info.get("device"):
            self.models_panel.rows[lm.spec.id].dot.setToolTip(f"Ready on {lm.info['device']}"
                                                              if lm.status == engine.READY else lm.error)
        if lm.info.get("fit_on") and lm.info["fit_on"] != "capnobase":
            self.toasts.show(f"{lm.spec.name}: development weights",
                             f"Its coefficients were fitted on {lm.info['fit_on']}, not CapnoBase, so its numbers "
                             "will not match Track 1. Refit with  setup.bat --zhu --capnobase PATH.")
        on_gpu = [m for m in self.models.values() if m.status == engine.READY and str(m.device).startswith("GPU")]
        self.s_device.setText(devmod.describe("gpu") if on_gpu else f"CPU: {devmod.cpu_name()}")
        ready = sum(m.status == engine.READY for m in self.models.values())
        loading = len(self.specs) - len(self.models)
        self.s_engine.setText(f"Models: {ready} ready" + (f", {loading} loading" if loading else ""))
        if lm.status == engine.READY and self.aligned is not None:
            self.rebuild_plots()

    # ================================================================ intake
    def _spinner(self):
        if self.pending:
            self.input.spinner.start()
        else:
            self.input.spinner.stop()

    def open_file(self, path, kind, rate=None, column=None, keep_first=False):
        if self.busy:
            self.toasts.show("Please wait", "The models are running; drop the file again when they finish.")
            return
        token = object()
        self.pending.add(token)
        QTimer.singleShot(config.SPINNER_DELAY_MS, self._spinner)

        def job():
            return intake.prepare(path, kind, self.settings, rate=rate, column=column, keep_first=keep_first)

        def done(prep):
            self.pending.discard(token)
            self._spinner()
            self._intake_done(kind, prep)

        def failed(e):
            self.pending.discard(token)
            self._spinner()
            self._intake_failed(path, kind, rate, column, keep_first, e)

        start(job, done, failed)

    def _intake_failed(self, path, kind, rate, column, keep_first, e):
        name = os.path.basename(path)
        what = "PPG" if kind == "ppg" else "true ECG"
        if isinstance(e, NeedRate):
            d = RateDialog(self, path)
            if d.exec() and d.rate:
                if d.remember.isChecked():
                    self.settings.remember_folder_rate(os.path.dirname(os.path.abspath(path)), d.rate)
                self.open_file(path, kind, d.rate, column, keep_first)
        elif isinstance(e, NeedColumn):
            col, ok = QInputDialog.getItem(self, "Which column?", f"Which column of {name} holds the {what}?",
                                           [str(c) for c in e.candidates], 0, False)
            if ok and col:
                self.open_file(path, kind, rate, col, keep_first)
        elif isinstance(e, TooLong):
            lim = f"{e.limit / 60:.0f} minutes"
            ans = popup(self, "Long file",
                        f"This file is {e.seconds / 60:.1f} min long. Files up to {lim} are processed in this "
                        "version (large-file mode, for recordings up to 2 h, comes later).\n\n"
                        f"Use its first {lim}?", buttons=(f"Use the first {lim}", "Choose another file"))
            if ans and ans.startswith("Use"):
                self.open_file(path, kind, rate, column, True)
        elif isinstance(e, IntakeError):
            popup(self, e.title, e.text)
        else:
            popup(self, "Cannot read file", f"{name}: {type(e).__name__}: {e}")

    def _intake_done(self, kind, prep):
        if kind == "ppg":
            if self.ecg_from_ppg_file and self.ecg is not None and self.ppg is not None and self.ecg.path != prep.path:
                self.ecg = None                      # the old PPG file's own ECG no longer belongs
                self.ecg_from_ppg_file = False
                self.input.show_prepared("ecg", None)
            self.ppg = prep
        else:
            self.ecg = prep
            self.ecg_from_ppg_file = False
        self.input.show_prepared(kind, prep)
        self.notify(prep.notices)
        self.inputs_changed()
        if kind == "ppg" and prep.embedded and self.ecg is None:
            name = next(iter(prep.embedded))
            keep = prep.cropped_from is not None

            def emb_done(e):
                if self.ppg is not prep or self.ecg is not None:
                    return
                self.ecg, self.ecg_from_ppg_file = e, True
                self.input.show_prepared("ecg", e)
                self.toasts.show("True ECG found in the same file",
                                 f"Column '{name}' is used as the true ECG. Remove it in the True ECG tab if you "
                                 "do not want errors measured against it.")
                self.notify([n for n in e.notices if n.level != "toast"])
                self.inputs_changed()

            start(lambda: intake.prepare_embedded(prep, name, keep_first=keep), emb_done,
                  lambda ex: self.toasts.show("Second signal not used", f"Column '{name}': {ex}"))

    def remove_input(self, kind):
        if self.busy:
            self.toasts.show("Please wait", "The models are running.")
            return
        if kind == "ppg":
            self.ppg = None
            if self.ecg_from_ppg_file:
                self.ecg, self.ecg_from_ppg_file = None, False
                self.input.show_prepared("ecg", None)
        else:
            self.ecg, self.ecg_from_ppg_file = None, False
        self.input.show_prepared(kind, None)
        self.inputs_changed()

    def notify(self, notices):
        for n in notices:
            if n.level == intake.MODAL:
                popup(self, n.title, n.text)
            elif n.level == intake.POPUP:
                popup(self, n.title, n.text, buttons=("Dismiss",), big=False, dim=False)
            else:
                self.toasts.show(n.title, n.text)

    def inputs_changed(self):
        had = bool(self.results)
        self.s_hover.setText("")
        self.entry += 1
        self.results = {}
        self.aligned = None
        self.model_src = None
        if self.ppg is not None:
            try:
                p, e, notes = intake.align(self.ppg, self.ecg)
            except IntakeError as ex:
                popup(self, ex.title, ex.text + "\n\nThe true ECG was removed.")
                self.ecg, self.ecg_from_ppg_file = None, False
                self.input.show_prepared("ecg", None)
                p, e, notes = intake.align(self.ppg, None)
            self.notify(notes)
            rp = np.array([], int)
            if e is not None:
                from bench import dsp
                rp = dsp.detect_rpeaks(dsp.bandpass(e, config.WORK_FS, 0.5, 40.0, order=2), config.WORK_FS)
            self.aligned = (p, e, intake.aligned_peaks(self.ppg, p), rp)
            self.model_src = intake.model_input(self.ppg, self.ecg if e is not None else None, p)
        for r in self.models_panel.rows.values():
            r.time.setText("" if r.status == engine.READY else r.time.text())
        self.models_panel.set_input(None if self.aligned is None else len(self.aligned[0]) / config.WORK_FS)
        self.plots.b_metrics.setEnabled(False)
        self.drawer.close_drawer()
        self.rebuild_plots()
        if had and self.aligned is not None:
            self.toasts.show("Inputs changed", "Press Run all to reconstruct the new input.")

    # ================================================================ plots
    def plot_rows(self):
        rows = [("ppg", "PPG", theme.PPG)]
        if self.aligned[1] is not None:
            rows.append(("ecg", "True ECG", theme.INK))
        for s in self.specs:
            lm = self.models.get(s.id)
            if lm is not None and lm.status == engine.READY:
                rows.append((s.id, s.name, theme.BLUE if s.id == "template" else theme.GREY))
        return rows

    def rebuild_plots(self):
        if self.aligned is None:
            self.plots.clear()
            return
        ppg, ecg, _, rp = self.aligned
        self.plots.start = float(self.models_panel.start.value())
        self.plots.length = float(self.models_panel.length.value())
        self.plots.build(len(ppg) / config.WORK_FS, self.plot_rows())
        self.redraw()

    def _ecg_display(self, ecg):
        y = band(ecg, self.plots.a_band.isChecked())
        if self.ecg is not None and self.ecg.inverted:
            y = -y
        return scale(y)

    def _recon_display(self, r):
        y = scale(band(r.pred, self.plots.a_band.isChecked()))
        if self.aligned[1] is not None and self.plots.a_align.isChecked():
            y = shift(y, display_shift(r.vs_true))
        return y

    def redraw(self):
        if self.aligned is None or not self.plots.rows:
            return
        ppg, ecg, _, rp = self.aligned
        self.plots.set_trace("ppg", scale(ppg), f"{len(ppg) / config.WORK_FS:.0f} s")
        if ecg is not None:
            self.plots.set_trace("ecg", self._ecg_display(ecg), "")
        for key in self.plots.rows:
            if key in ("ppg", "ecg"):
                continue
            r = self.results.get(key)
            if r is None:
                self.plots.set_trace(key, None, "", "not run yet: press ▶ or Run all")
            else:
                self.plots.set_trace(key, self._recon_display(r), f"{r.gen_s:.2f} s")
        self.plots.set_rpeaks(rp)

    # ================================================================ running
    def run(self, ids):
        if self.aligned is None or self.busy:
            return
        ids = [i for i in ids if i in self.models and self.models[i].status == engine.READY]
        if not ids:
            return
        ids = engine.run_order(ids, self.specs)          # fast models first, RDDM and PPGFlowECG last
        entry = self.entry
        ppg, ecg, peaks, _ = self.aligned
        src = self.model_src
        self.busy = True
        self.models_panel.set_busy(True)
        t0 = time.perf_counter()
        models = [self.models[i] for i in ids]

        def job():
            for lm in models:
                in_gui(self._model_started, (entry, lm.spec.id))
                try:
                    r = self.engine.run(lm.spec.id, ppg, peaks, ecg, src=src).result()
                except EngineCrashed as ex:
                    in_gui(self._model_failed, (entry, lm.spec.id, ex))
                    break                                # the engine restarts; the rest would fail too
                except Exception as ex:
                    in_gui(self._model_failed, (entry, lm.spec.id, ex))
                    continue
                in_gui(self._model_result, (entry, r))
            return entry, time.perf_counter() - t0, len(models)

        start(job, self._run_finished, lambda ex: self._run_finished((entry, 0.0, 0), ex))

    def _model_started(self, arg):
        entry, mid = arg
        if entry == self.entry:
            self.models_panel.rows[mid].set_running(True)

    def _model_failed(self, arg):
        entry, mid, ex = arg
        row = self.models_panel.rows[mid]
        row.set_running(False)
        row.time.setText("error")
        if entry == self.entry:
            self.toasts.show(f"{row.spec.name} failed", f"{type(ex).__name__}: {ex}"[:240])

    def _model_result(self, arg):
        entry, r = arg
        row = self.models_panel.rows[r.model_id]
        row.set_running(False)
        if entry != self.entry:
            return
        self.results[r.model_id] = r
        row.set_time(r.gen_s)
        if r.model_id in self.plots.rows:
            self.plots.set_trace(r.model_id, self._recon_display(r), f"{r.gen_s:.2f} s")
        self.plots.b_metrics.setEnabled(True)
        if self.drawer.isVisible():
            self.update_drawer()

    def _run_finished(self, arg, error=None):
        entry, secs, n = arg
        self.busy = False
        self.models_panel.set_busy(False)
        for row in self.models_panel.rows.values():
            row.set_running(False)
        if error is not None:
            popup(self, "Run stopped", f"{type(error).__name__}: {error}")
            return
        if entry != self.entry:
            return
        self.statusBar().showMessage(f"{n} model{'s' if n != 1 else ''} ran in {secs:.2f} s", 6000)
        if self.aligned[1] is None and self.told_no_ecg != entry and self.results:
            self.told_no_ecg = entry
            self.toasts.show("No true ECG given", "Errors against a true ECG are skipped. Add one in the True ECG "
                                                  "tab to see them.")
        if self.results and not self.drawer.isVisible() and n > 1:
            self.toggle_metrics()

    # ================================================================ metrics
    def update_drawer(self):
        order = [s.id for s in self.specs]
        res = [self.results[i] for i in order if i in self.results]
        sub = f"{self.ppg.name} · {fmt_duration(len(self.aligned[0]) / config.WORK_FS)}" if self.ppg else ""
        self.drawer.update_results(res, self.aligned is not None and self.aligned[1] is not None, sub)

    def toggle_metrics(self):
        if not self.results:
            return
        self.update_drawer()
        self.drawer.toggle()
