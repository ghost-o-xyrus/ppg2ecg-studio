"""Bottom half: stacked traces on one time axis (PPG, true ECG, every model) and the icon strip."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMenu, QToolButton, QVBoxLayout, QWidget

from .. import config
from . import theme

LABEL_W = 150
TIME_W = 70


def scale(y):
    """Robust scaling to about [-1, 1] (percentiles, so one artefact does not flatten the trace)."""
    y = np.asarray(y, float)
    if len(y) == 0 or np.ptp(y) < 1e-12:
        return np.zeros_like(y)
    lo, hi = np.percentile(y, [0.5, 99.5])
    if hi - lo < 1e-12:
        lo, hi = y.min(), y.max()
    return (y - (lo + hi) / 2) / ((hi - lo) / 2)


def band(y, on=True):
    from bench import dsp
    y = np.asarray(y, float)
    if not on or len(y) < 64 or np.ptp(y) < 1e-12:
        return y
    return dsp.bandpass(y, config.WORK_FS, 0.5, 40.0, order=2)


def shift(y, k):
    """Move a trace k samples earlier (k > 0) keeping its length; the gap is left empty (NaN)."""
    if k == 0:
        return y
    out = np.full_like(y, np.nan)
    if k > 0:
        out[:-k] = y[k:]
    else:
        out[-k:] = y[:k]
    return out


class Row:
    def __init__(self, key, name, color, plot, label, tlabel):
        self.key, self.name, self.color = key, name, color
        self.plot, self.label, self.tlabel = plot, label, tlabel
        self.curve = plot.plot(pen=pg.mkPen(color, width=1.15))
        self.curve.setClipToView(True)
        self.curve.setDownsampling(auto=True, method="peak")
        self.rlines = pg.PlotCurveItem(pen=pg.mkPen(theme.RPEAK, width=0.8, style=Qt.DashLine), connect="pairs")
        self.rlines.setZValue(-10)
        plot.addItem(self.rlines)
        self.empty = pg.TextItem("", color=theme.MUTED, anchor=(0, 0.5))
        plot.addItem(self.empty, ignoreBounds=True)
        self.y = None

    def set(self, y, fs):
        self.y = y
        if y is None:
            self.curve.setData([], [])
        else:
            self.curve.setData(np.arange(len(y)) / fs, y, connect="finite")


class PlotArea(QFrame):
    showMetrics = Signal()
    optionsChanged = Signal()
    hover = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        pg.setConfigOptions(antialias=True, background="w", foreground=theme.MUTED)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 8, 6, 8)
        outer.setSpacing(4)
        self.stage = QWidget()                      # plots + the metrics drawer overlay
        st = QVBoxLayout(self.stage)
        st.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.ci.setSpacing(0)
        self.placeholder = QLabel("Drop a PPG file in the Input panel, then press Run all.\n"
                                  "The PPG, the true ECG (if given) and every reconstruction appear here, one "
                                  "under the other.")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setObjectName("muted")
        st.addWidget(self.placeholder)
        st.addWidget(self.glw)
        self.glw.hide()
        outer.addWidget(self.stage, 1)

        strip = QVBoxLayout()
        strip.setSpacing(6)
        self.b_metrics = self._strip("📄", "See metrics")
        self.b_compare = self._strip("⇆", "Compare with a previous entry (arrives in a later version)")
        self.b_export = self._strip("💾", "Export (arrives in a later version)")
        self.b_options = self._strip("⚙", "Display options")
        self.b_metrics.clicked.connect(self.showMetrics.emit)
        self.b_compare.setEnabled(False)
        self.b_export.setEnabled(False)
        self.b_metrics.setEnabled(False)
        menu = QMenu(self)
        self.a_align = QAction("Align to true ECG (remove each model's constant delay)", self, checkable=True,
                               checked=True)
        self.a_rpeaks = QAction("Show true R peaks", self, checkable=True, checked=True)
        self.a_band = QAction("Band-pass 0.5-40 Hz (as scored)", self, checkable=True, checked=True)
        for a in (self.a_align, self.a_rpeaks, self.a_band):
            menu.addAction(a)
            a.toggled.connect(lambda _=False: self.optionsChanged.emit())
        self.b_options.setMenu(menu)
        self.b_options.setPopupMode(QToolButton.InstantPopup)
        for b in (self.b_metrics, self.b_compare, self.b_export, self.b_options):
            strip.addWidget(b)
        strip.addStretch(1)
        outer.addLayout(strip)

        self.rows: dict[str, Row] = {}
        self.fs = config.WORK_FS
        self.duration = 0.0
        self.start, self.length = 0.0, float(config.PLOT_DEFAULT_S)
        self.rpeaks = np.array([], int)
        self.glw.scene().sigMouseMoved.connect(self._moved)
        self.glw.scene().sigMouseClicked.connect(self._clicked)

    def _strip(self, icon, tip):
        b = QToolButton(text=icon)
        b.setObjectName("strip")
        b.setToolTip(tip)
        b.setFixedSize(42, 42)
        return b

    # ------------------------------------------------------------ building
    def clear(self):
        self.glw.ci.clear()
        self.rows = {}
        self.glw.hide()
        self.placeholder.show()

    def build(self, duration, rows):
        """rows: list of (key, name, colour). Rebuilds the stack."""
        self.glw.ci.clear()
        self.rows = {}
        self.duration = duration
        first = None
        n = len(rows)
        for i, (key, name, color) in enumerate(rows):
            lab = self.glw.ci.addLabel(f"<span style='color:{color}; font-weight:600'>{name}</span>", row=i, col=0,
                                       justify="right")
            p = self.glw.ci.addPlot(row=i, col=1)
            tl = self.glw.ci.addLabel("", row=i, col=2, justify="left", color=theme.MUTED, size="9pt")
            p.hideAxis("left")
            p.showAxis("bottom", i == n - 1)
            if i == n - 1:
                p.setLabel("bottom", "time (s)")
            p.setMenuEnabled(False)
            p.hideButtons()
            p.setMouseEnabled(x=True, y=False)
            p.setYRange(-1.25, 1.25, padding=0)
            p.setLimits(xMin=0, xMax=max(duration, 1.0), minXRange=0.5, maxXRange=max(duration, 1.0))
            p.getViewBox().setDefaultPadding(0)
            if first is None:
                first = p
            else:
                p.setXLink(first)
            self.rows[key] = Row(key, name, color, p, lab, tl)
        lay = self.glw.ci.layout
        lay.setColumnFixedWidth(0, LABEL_W)
        lay.setColumnFixedWidth(2, TIME_W)
        self.placeholder.hide()
        self.glw.show()
        self.set_window(self.start, self.length)

    def set_trace(self, key, y, right_text="", empty_text=""):
        r = self.rows.get(key)
        if r is None:
            return
        r.set(y, self.fs)
        r.tlabel.setText(right_text, color=theme.MUTED, size="9pt")
        r.empty.setText(empty_text if y is None else "")
        r.empty.setPos(self.start + 0.1, 0)

    def set_rpeaks(self, peaks):
        self.rpeaks = np.asarray(peaks, int)
        self._draw_rpeaks()

    def _draw_rpeaks(self):
        on = self.a_rpeaks.isChecked() and len(self.rpeaks) > 0
        t = self.rpeaks / self.fs
        x = np.repeat(t, 2)
        y = np.tile([-1.25, 1.25], len(t))
        for r in self.rows.values():
            if on:
                r.rlines.setData(x, y)
            else:
                r.rlines.setData([], [])

    def refresh_options(self):
        self._draw_rpeaks()

    # ------------------------------------------------------------ window and mouse
    def set_window(self, start, length):
        self.start, self.length = float(start), float(length)
        if not self.rows:
            return
        end = min(self.start + self.length, max(self.duration, self.start + 0.5))
        p = next(iter(self.rows.values())).plot
        p.setXRange(self.start, end, padding=0)
        for r in self.rows.values():
            r.empty.setPos(self.start + 0.1, 0)

    def _clicked(self, ev):
        if ev.double():
            self.set_window(self.start, self.length)

    def _moved(self, pos):
        for r in self.rows.values():
            if r.plot.sceneBoundingRect().contains(pos):
                pt = r.plot.getViewBox().mapSceneToView(pos)
                t = pt.x()
                txt = f"t = {t:.3f} s"
                if r.y is not None:
                    i = int(round(t * self.fs))
                    if 0 <= i < len(r.y) and np.isfinite(r.y[i]):
                        txt += f"   {r.name}: {r.y[i]:+.3f}"
                self.hover.emit(txt)
                return
        self.hover.emit("")
