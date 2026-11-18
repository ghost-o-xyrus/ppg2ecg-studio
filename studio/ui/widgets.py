"""Small widgets shared by the window: background tasks, status dot, drop zone, file card, spinner, side
notifications, central pop-ups and the sampling-rate question."""
from __future__ import annotations

import os
import traceback

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import (QEasingCurve, QEvent, QObject, QPropertyAnimation, QRectF, QRunnable, QSize, Qt, QThreadPool,
                            QTimer, Signal, Slot)
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QFrame, QGraphicsOpacityEffect,
                               QHBoxLayout, QLabel, QPushButton, QSizePolicy, QToolButton, QVBoxLayout, QWidget)

from .. import config
from . import theme


# ================================================================ background tasks
class _Dispatcher(QObject):
    """Lives in the GUI thread; worker threads emit `call` and the callback runs in the GUI thread."""
    call = Signal(object, object)

    def __init__(self):
        super().__init__()
        self.call.connect(self._run)

    @Slot(object, object)
    def _run(self, fn, arg):
        try:
            fn(arg)
        except Exception:
            traceback.print_exc()


_DISPATCH = None


def dispatcher() -> _Dispatcher:
    global _DISPATCH
    if _DISPATCH is None:
        _DISPATCH = _Dispatcher()
    return _DISPATCH


def in_gui(fn, arg=None):
    """Call fn(arg) in the GUI thread (safe from any thread). Dropped quietly once the app is closing."""
    try:
        dispatcher().call.emit(fn, arg)
    except RuntimeError:            # the window closed while a background job was still running
        pass


class Task(QRunnable):
    def __init__(self, fn, on_done=None, on_error=None):
        super().__init__()
        self.fn, self.on_done, self.on_error = fn, on_done, on_error
        self.setAutoDelete(True)

    def run(self):
        try:
            r = self.fn()
        except BaseException as e:      # reported in the GUI, never kills the app
            if not isinstance(e, Exception):
                raise
            if self.on_error:
                in_gui(self.on_error, e)
            else:
                traceback.print_exc()
            return
        if self.on_done:
            in_gui(self.on_done, r)


def start(fn, on_done=None, on_error=None):
    dispatcher()            # make sure it is created in the GUI thread
    QThreadPool.globalInstance().start(Task(fn, on_done, on_error))


# ================================================================ status dot
class StatusDot(QWidget):
    COLORS = {"loading": theme.HOLLOW, "ready": theme.GREEN, "failed": theme.RED, "not installed": None,
              "running": theme.AMBER}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self.state = "loading"
        self._phase = 0.0
        self._timer = QTimer(self, interval=60, timeout=self._tick)

    def set_state(self, state, tip=""):
        self.state = state
        self.setToolTip(tip or state.capitalize())
        if state in ("loading", "running"):
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def _tick(self):
        self._phase = (self._phase + 0.08) % 1.0
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = self.COLORS.get(self.state)
        r = QRectF(2, 2, 10, 10)
        if c is None:
            p.setPen(QPen(QColor(theme.HOLLOW), 1.5))
            p.drawEllipse(r)
            return
        col = QColor(c)
        if self.state in ("loading", "running"):
            col.setAlphaF(0.45 + 0.55 * abs(np.sin(np.pi * self._phase)))
        p.setPen(Qt.NoPen)
        p.setBrush(col)
        p.drawEllipse(r)


# ================================================================ drop zone and file card
def _local_file(event):
    md = event.mimeData()
    if md.hasUrls():
        for u in md.urls():
            if u.isLocalFile() and os.path.isfile(u.toLocalFile()):
                return u.toLocalFile()
    return None


FILE_FILTER = "Signals (*.csv *.txt *.tsv *.dat *.npy *.npz);;All files (*)"


class DropZone(QFrame):
    fileChosen = Signal(str)

    def __init__(self, kind, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(150)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        what = "PPG" if kind == "ppg" else "true ECG"
        icon = QLabel("⤓")
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet(f"font-size: 26pt; color: {theme.BLUE};")
        row = QHBoxLayout()
        row.setAlignment(Qt.AlignCenter)
        row.addWidget(QLabel(f"Drop your {what} file here, or"))
        b = QPushButton("Browse…")
        b.setObjectName("link")
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(self.browse)
        row.addWidget(b)
        rules = QLabel(f"CSV, TXT, NPY or NPZ · {config.MIN_SECONDS:.0f} s to {config.MAX_SECONDS / 60:.0f} min · "
                       f"up to {config.MAX_MB:.0f} MB")
        rules.setObjectName("small")
        rules.setAlignment(Qt.AlignCenter)
        lay.addWidget(icon)
        lay.addLayout(row)
        lay.addWidget(rules)
        if kind == "ecg":
            hint = QLabel("Optional. With a true ECG, every reconstruction is also scored against it.")
            hint.setObjectName("small")
            hint.setAlignment(Qt.AlignCenter)
            hint.setWordWrap(True)
            lay.addWidget(hint)

    def browse(self):
        what = "PPG" if self.kind == "ppg" else "true ECG"
        f, _ = QFileDialog.getOpenFileName(self, f"Open {what} file", "", FILE_FILTER)
        if f:
            self.fileChosen.emit(f)

    def _hover(self, on):
        self.setProperty("hover", "true" if on else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, e):
        if _local_file(e):
            e.acceptProposedAction()
            self._hover(True)

    def dragLeaveEvent(self, e):
        self._hover(False)

    def dropEvent(self, e):
        self._hover(False)
        f = _local_file(e)
        if f:
            e.acceptProposedAction()
            self.fileChosen.emit(f)


class Sparkline(pg.PlotWidget):
    def __init__(self, color, parent=None):
        super().__init__(parent, background=None)
        self.setFixedHeight(46)
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setMouseEnabled(False, False)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.getPlotItem().setContentsMargins(0, 0, 0, 0)
        self.curve = self.plot(pen=pg.mkPen(color, width=1.2))
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def set_data(self, y, fs, seconds=10.0):
        y = np.asarray(y[: int(seconds * fs)], float)
        self.curve.setData(np.arange(len(y)) / fs, y)
        self.enableAutoRange()


def fmt_duration(s):
    m, sec = divmod(int(round(s)), 60)
    return f"{m}:{sec:02d} min" if m else f"{s:.1f} s"


class FileCard(QFrame):
    removed = Signal()
    fileChosen = Signal(str)

    def __init__(self, kind, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.setObjectName("fileCard")
        self.setAcceptDrops(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 10, 10)
        top = QHBoxLayout()
        self.name = QLabel()
        self.name.setStyleSheet("font-weight: 600; font-size: 10.5pt;")
        self.name.setTextInteractionFlags(Qt.TextSelectableByMouse)
        x = QToolButton(text="✕")
        x.setObjectName("close")
        x.setToolTip("Remove this file")
        x.clicked.connect(self.removed.emit)
        top.addWidget(self.name, 1)
        top.addWidget(x)
        self.facts = QLabel()
        self.facts.setObjectName("muted")
        self.facts.setWordWrap(True)
        self.spark = Sparkline(theme.PPG if kind == "ppg" else theme.INK)
        hint = QLabel("Drop another file here to replace it.")
        hint.setObjectName("small")
        lay.addLayout(top)
        lay.addWidget(self.facts)
        lay.addWidget(self.spark)
        lay.addWidget(hint)

    def show_prepared(self, prep):
        self.name.setText(prep.name)
        beats = "beats" if prep.kind == "ppg" else "R peaks"
        parts = [f"Length {fmt_duration(prep.seconds)}",
                 f"{prep.fs_in:g} Hz ({prep.fs_source}) → 128 Hz",
                 f"{prep.n_beats} {beats} found"]
        if prep.column and prep.column != "array" and not prep.column.startswith("column "):
            parts.append(f"column '{prep.column}'")
        if prep.windows_shape:
            parts.append(f"{prep.windows_shape[0]} windows joined")
        if prep.flipped:
            parts.append("flipped upright")
        if prep.cropped_from:
            parts.append(f"first {config.MAX_SECONDS / 60:.0f} min of {fmt_duration(prep.cropped_from)}")
        self.facts.setText(" · ".join(parts))
        y = -prep.x if prep.inverted else prep.x
        self.spark.set_data(y, prep.fs)
        self.setToolTip(prep.path)

    def dragEnterEvent(self, e):
        if _local_file(e):
            e.acceptProposedAction()

    def dropEvent(self, e):
        f = _local_file(e)
        if f:
            e.acceptProposedAction()
            self.fileChosen.emit(f)


# ================================================================ spinner overlay
class Spinner(QWidget):
    """Covers its parent with a translucent layer, a turning arc and a line of text."""

    def __init__(self, parent, text="Adjusting data to preferred format…"):
        super().__init__(parent)
        self.text = text
        self.angle = 0
        self.timer = QTimer(self, interval=16, timeout=self._tick)
        self.hide()

    def _tick(self):
        self.angle = (self.angle + 6) % 360
        self.update()

    def start(self):
        self.setGeometry(self.parentWidget().rect())
        self.raise_()
        self.show()
        self.timer.start()

    def stop(self):
        self.timer.stop()
        self.hide()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(255, 255, 255, 240))
        c = self.rect().center()
        r = QRectF(c.x() - 18, c.y() - 34, 36, 36)
        p.setPen(QPen(QColor(theme.LINE), 4))
        p.drawEllipse(r)
        pen = QPen(QColor(theme.BLUE), 4)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(r, -self.angle * 16, 100 * 16)
        p.setPen(QColor(theme.INK))
        f = p.font()
        f.setPointSizeF(10.5)
        p.setFont(f)
        p.drawText(QRectF(0, c.y() + 12, self.width(), 24), Qt.AlignHCenter | Qt.AlignTop, self.text)


# ================================================================ side notifications
class Toast(QFrame):
    closed = Signal(object)

    def __init__(self, title, text, parent=None, seconds=8):
        super().__init__(parent)
        self.title = title
        self.setObjectName("toast")
        self.setFixedWidth(340)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 10, 6, 10)
        col = QVBoxLayout()
        t = QLabel(title)
        t.setObjectName("toastTitle")
        b = QLabel(text)
        b.setWordWrap(True)
        col.addWidget(t)
        col.addWidget(b)
        x = QToolButton(text="✕")
        x.clicked.connect(self.close_now)
        lay.addLayout(col, 1)
        lay.addWidget(x, 0, Qt.AlignTop)
        self.eff = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.eff)
        self.anim = QPropertyAnimation(self.eff, b"opacity", self)
        self.anim.setDuration(180)
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)
        self.anim.start()
        self._timer = QTimer(self, singleShot=True, interval=int(seconds * 1000), timeout=self.close_now)
        self._timer.start()
        self._closing = False

    def close_now(self):
        if self._closing:
            return
        self._closing = True
        self.hide()
        self.closed.emit(self)
        self.deleteLater()


class ToastHost(QObject):
    """Stacks side notifications in the bottom-right corner of `host`."""

    def __init__(self, host: QWidget, margin=16):
        super().__init__(host)
        self.host, self.margin = host, margin
        self.toasts: list = []
        host.installEventFilter(self)

    def show(self, title, text, seconds=8):
        for old in [t for t in self.toasts if t.title == title]:   # the same notice again replaces the old one
            old.close_now()
        t = Toast(title, text, self.host, seconds)
        t.closed.connect(self._remove)
        self.toasts.append(t)
        t.show()
        t.adjustSize()
        self.layout()
        while len(self.toasts) > 5:
            self.toasts[0].close_now()

    def _remove(self, t):
        if t in self.toasts:
            self.toasts.remove(t)
        self.layout()

    def layout(self):
        y = self.host.height() - self.margin
        for t in reversed(self.toasts):
            t.adjustSize()
            y -= t.height()
            t.move(self.host.width() - t.width() - self.margin, y)
            t.raise_()
            y -= 8

    def eventFilter(self, obj, ev):
        if obj is self.host and ev.type() == QEvent.Type.Resize:
            self.layout()
        return False


# ================================================================ central pop-ups
class Dimmer(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setGeometry(parent.rect())
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)

    def paintEvent(self, _):
        QPainter(self).fillRect(self.rect(), QColor(17, 24, 39, 110))


def popup(parent, title, text, buttons=("OK",), big=True, dim=True, danger=None):
    """Central pop-up. Returns the label of the button pressed (or None if closed)."""
    dimmer = None
    if dim and parent is not None:
        dimmer = Dimmer(parent.window())
        dimmer.show()
        dimmer.raise_()
    d = QDialog(parent)
    d.setObjectName("popup")
    d.setWindowTitle(title)
    d.setModal(True)
    lay = QVBoxLayout(d)
    lay.setContentsMargins(28, 24, 28, 20)
    lay.setSpacing(14)
    t = QLabel(title)
    t.setObjectName("popupTitle" if big else "panelTitle")
    m = QLabel(text)
    m.setObjectName("popupText" if big else "")
    m.setWordWrap(True)
    m.setMinimumWidth(460 if big else 380)
    lay.addWidget(t)
    lay.addWidget(m)
    row = QHBoxLayout()
    row.addStretch(1)
    chosen = {"v": None}
    for i, label in enumerate(buttons):
        b = QPushButton(label)
        if label == danger:
            b.setObjectName("danger")
        elif i == 0:
            b.setObjectName("primary")
            b.setDefault(True)
        b.clicked.connect(lambda _=False, lab=label: (chosen.__setitem__("v", lab), d.accept()))
        row.addWidget(b)
    lay.addLayout(row)
    d.exec()
    if dimmer is not None:
        dimmer.deleteLater()
    return chosen["v"]


class RateDialog(QDialog):
    """Asks for a sampling rate the file does not state; remembered for files from the same folder."""

    def __init__(self, parent, path, guess=None):
        super().__init__(parent)
        self.setWindowTitle("Sampling rate")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 16)
        t = QLabel(f"At what rate was <b>{os.path.basename(path)}</b> recorded?")
        t.setWordWrap(True)
        sub = QLabel("The file does not say (no time column, no '# fs=' line, no '_125Hz' in its name).")
        sub.setObjectName("small")
        sub.setWordWrap(True)
        row = QHBoxLayout()
        self.box = QComboBox()
        self.box.setEditable(True)
        self.box.addItems([str(r) for r in config.COMMON_RATES])
        self.box.setCurrentText(str(int(guess)) if guess else "125")
        self.box.setMinimumWidth(110)
        row.addWidget(self.box)
        row.addWidget(QLabel("Hz"))
        row.addStretch(1)
        self.remember = QCheckBox("Use this rate for other files from the same folder")
        self.remember.setChecked(True)
        self.err = QLabel("")
        self.err.setStyleSheet(f"color: {theme.RED};")
        btns = QHBoxLayout()
        btns.addStretch(1)
        ok = QPushButton("Use this rate")
        ok.setObjectName("primary")
        ok.setDefault(True)
        cancel = QPushButton("Cancel")
        ok.clicked.connect(self._ok)
        cancel.clicked.connect(self.reject)
        btns.addWidget(cancel)
        btns.addWidget(ok)
        for w in (t, sub):
            lay.addWidget(w)
        lay.addLayout(row)
        lay.addWidget(self.remember)
        lay.addWidget(self.err)
        lay.addLayout(btns)
        self.rate = None

    def _ok(self):
        try:
            v = float(self.box.currentText().replace(",", ".").strip())
        except ValueError:
            v = 0
        if not (1 <= v <= 100000):
            self.err.setText("Enter a rate in Hz, for example 125.")
            return
        self.rate = v
        self.accept()


def hline():
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"color: {theme.LINE};")
    return f


def expanding(w):
    w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return w


__all__ = ["start", "in_gui", "StatusDot", "DropZone", "FileCard", "Spinner", "ToastHost", "popup", "RateDialog",
           "QSize", "QEasingCurve"]
