"""Top right: the models. Run all, the plot window (length and start), and one row per model."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QScrollArea, QSpinBox, QToolButton, QVBoxLayout, QWidget)

from .. import config
from ..engine import FAILED, LOADING, MISSING, READY
from .widgets import StatusDot

NEED_PPG = "Drop a PPG file first"


class ModelRow(QFrame):
    runClicked = Signal(str)

    def __init__(self, spec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.setObjectName("modelRow")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 6, 4, 6)
        lay.setSpacing(8)
        self.check = QCheckBox()
        self.check.setChecked(True)
        self.check.setToolTip("Include in Run all")
        self.name = QLabel(spec.name)
        self.name.setStyleSheet("font-weight: 600;")
        tip = spec.description + (f"\n\n{spec.citation}" if spec.citation else "")
        self.name.setToolTip(tip)
        self.tag = QLabel(spec.family or "model")
        self.tag.setObjectName("tag")
        self.dot = StatusDot()
        self.time = QLabel("")
        self.time.setObjectName("muted")
        self.time.setMinimumWidth(64)
        self.time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)               # busy indicator while the model runs
        self.bar.setFixedWidth(70)
        self.bar.setTextVisible(False)
        self.bar.hide()
        self.play = QToolButton(text="▶")
        self.play.setObjectName("play")
        self.play.clicked.connect(lambda: self.runClicked.emit(self.spec.id))
        lay.addWidget(self.check)
        lay.addWidget(self.dot)
        lay.addWidget(self.name)
        lay.addWidget(self.tag)
        lay.addStretch(1)
        lay.addWidget(self.bar)
        lay.addWidget(self.time)
        lay.addWidget(self.play)
        self.status = LOADING
        self.set_status(LOADING)

    def set_status(self, status, detail=""):
        self.status = status
        tips = {LOADING: "Loading…", READY: "Ready", FAILED: f"Failed to load: {detail}",
                MISSING: f"Not installed: {detail}\nInstall it with  setup.bat --{self.spec.id.split('_')[0]}"}
        self.dot.set_state(status, tips.get(status, status))
        off = status in (MISSING, FAILED)
        self.setProperty("off", "true" if off else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        if off:
            self.check.setChecked(False)
            self.time.setText("not installed" if status == MISSING else "failed")
        elif self.time.text() in ("not installed", "failed"):
            self.time.setText("")
        self.check.setEnabled(not off)

    def set_running(self, on):
        self.bar.setVisible(on)
        self.dot.set_state("running" if on else self.status, "Running…" if on else "Ready")
        if on:
            self.time.setText("")

    def set_time(self, seconds):
        self.time.setText(f"{seconds:.2f} s" if seconds >= 0.01 else f"{seconds * 1000:.0f} ms")
        self.time.setToolTip("Generation time of the last run")


class ModelPanel(QFrame):
    runAll = Signal()
    runOne = Signal(str)
    windowChanged = Signal(float, float)      # start, length (s)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        title = QLabel("Models")
        title.setObjectName("panelTitle")
        bar = QHBoxLayout()
        self.run_all = QPushButton("▶  Run all")
        self.run_all.setObjectName("primary")
        self.run_all.clicked.connect(self.runAll.emit)
        self.length = QSpinBox()
        self.length.setRange(config.PLOT_MIN_S, config.PLOT_MAX_S)
        self.length.setValue(config.PLOT_DEFAULT_S)
        self.length.setSuffix(" s")
        self.length.setToolTip(f"Plot length ({config.PLOT_MIN_S}-{config.PLOT_MAX_S} s). Models always run on the "
                               "whole file.")
        self.start = QDoubleSpinBox()
        self.start.setRange(0, 0)
        self.start.setDecimals(1)
        self.start.setSingleStep(1.0)
        self.start.setSuffix(" s")
        self.start.setToolTip("Start time of the plotted window")
        self.length.valueChanged.connect(self._emit_window)
        self.start.valueChanged.connect(self._emit_window)
        bar.addWidget(self.run_all)
        bar.addSpacing(14)
        bar.addWidget(QLabel("Show"))
        bar.addWidget(self.length)
        bar.addWidget(QLabel("from"))
        bar.addWidget(self.start)
        bar.addStretch(1)
        lay.addWidget(title)
        lay.addLayout(bar)
        self.list = QWidget()
        self.list_lay = QVBoxLayout(self.list)
        self.list_lay.setContentsMargins(0, 0, 0, 0)
        self.list_lay.setSpacing(0)
        self.list_lay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.list)
        lay.addWidget(scroll, 1)
        self.foot = QLabel("")
        self.foot.setObjectName("small")
        self.foot.setWordWrap(True)
        lay.addWidget(self.foot)
        self.rows: dict[str, ModelRow] = {}
        self.has_input = False
        self.busy = False

    def set_models(self, specs):
        for s in specs:
            r = ModelRow(s)
            r.runClicked.connect(self.runOne.emit)
            r.check.toggled.connect(lambda _=False: self.refresh())
            self.rows[s.id] = r
            self.list_lay.insertWidget(self.list_lay.count() - 1, r)
        self.refresh()

    def set_status(self, model_id, status, detail=""):
        self.rows[model_id].set_status(status, detail)
        missing = [r.spec.name for r in self.rows.values() if r.status == MISSING]
        self.foot.setText(("Not installed: " + ", ".join(missing) + ". Run setup.bat to add them.")
                          if missing else "")
        self.refresh()

    def selected(self):
        return [i for i, r in self.rows.items() if r.check.isChecked() and r.status == READY]

    def set_input(self, seconds):
        """seconds = length of the loaded PPG, or None when there is none."""
        self.has_input = seconds is not None
        self.start.blockSignals(True)
        self.start.setRange(0, max(0.0, (seconds or 0) - config.PLOT_MIN_S))
        self.start.setValue(0)
        self.start.blockSignals(False)
        self.refresh()

    def set_busy(self, busy):
        self.busy = busy
        self.refresh()

    def refresh(self):
        ok = self.has_input and not self.busy
        n = len(self.selected())
        self.run_all.setEnabled(ok and n > 0)
        self.run_all.setToolTip(NEED_PPG if not self.has_input else
                                "Running…" if self.busy else
                                "Tick at least one ready model" if n == 0 else f"Run the {n} ticked models")
        for w in (self.length, self.start):
            w.setEnabled(self.has_input)
            if not self.has_input:
                w.setToolTip(NEED_PPG)
        for r in self.rows.values():
            ready = r.status == READY
            r.play.setEnabled(ok and ready)
            r.play.setToolTip("Run" if ok and ready else NEED_PPG if not self.has_input else
                              "Running…" if self.busy else "This model is not ready")

    def _emit_window(self, *_):
        self.windowChanged.emit(float(self.start.value()), float(self.length.value()))
