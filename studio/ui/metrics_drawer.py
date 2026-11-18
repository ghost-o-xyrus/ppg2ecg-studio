"""Metrics drawer: slides over the right 40% of the plots. Tabs: Model comparison | Errors vs true ECG."""
from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QEvent, QPoint, QPropertyAnimation, Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (QAbstractItemView, QFrame, QHBoxLayout, QHeaderView, QLabel, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout, QWidget)

from .. import scoring
from . import theme

NO_ECG = ("No true ECG was given, so there is nothing to measure errors against.\n"
          "Add one in the True ECG tab and run again.")


def make_table():
    t = QTableWidget()
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionMode(QAbstractItemView.NoSelection)
    t.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
    t.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
    t.setAlternatingRowColors(False)
    t.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    t.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    t.setWordWrap(False)
    return t


def fill(table, columns, results):
    table.clear()
    table.setColumnCount(len(columns))
    table.setRowCount(len(results))
    for j, c in enumerate(columns):
        h = QTableWidgetItem(c.label + (f" ({c.unit})" if c.unit else ""))
        h.setToolTip(c.tip or c.label)
        table.setHorizontalHeaderItem(j, h)
    table.setVerticalHeaderLabels([r.model_name for r in results])
    rows = [r.row(columns) for r in results]
    bold = QFont()
    bold.setBold(True)
    for j, c in enumerate(columns):
        best = scoring.best_index([row[j] for row in rows], c.better)
        for i, row in enumerate(rows):
            it = QTableWidgetItem(scoring.fmt(row[j], c))
            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter if not isinstance(row[j], str) else
                                Qt.AlignLeft | Qt.AlignVCenter)
            if best == i:
                it.setFont(bold)
                it.setToolTip("Best in this column")
            if results[i].model_id == "template":
                it.setForeground(QBrush(QColor(theme.BLUE)))
            table.setItem(i, j, it)
    table.resizeColumnsToContents()


class MetricsDrawer(QFrame):
    def __init__(self, stage: QWidget):
        super().__init__(stage)
        self.stage = stage
        self.setObjectName("drawer")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 8, 10)
        head = QHBoxLayout()
        t = QLabel("Metrics")
        t.setObjectName("panelTitle")
        self.sub = QLabel("")
        self.sub.setObjectName("small")
        x = QToolButton(text="✕")
        x.setObjectName("close")
        x.setToolTip("Close")
        x.clicked.connect(self.close_drawer)
        head.addWidget(t)
        head.addWidget(self.sub, 1)
        head.addWidget(x)
        lay.addLayout(head)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.t_cmp = make_table()
        self.t_err = make_table()
        self.err_stack = QStackedWidget()
        msg = QLabel(NO_ECG)
        msg.setWordWrap(True)
        msg.setAlignment(Qt.AlignCenter)
        msg.setObjectName("muted")
        self.err_stack.addWidget(self.t_err)
        self.err_stack.addWidget(msg)
        self.tabs.addTab(self.t_cmp, "Model comparison")
        self.tabs.addTab(self.err_stack, "Errors vs true ECG")
        lay.addWidget(self.tabs, 1)
        note = QLabel("Bold = best in the column. Errors use the Track 1 scorer (10-s windows, delay-corrected "
                      "R peaks). Hover a header for its meaning.")
        note.setObjectName("small")
        note.setWordWrap(True)
        lay.addWidget(note)
        self.anim = QPropertyAnimation(self, b"pos", self)
        self.anim.setDuration(220)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        self.hide()
        self.stage.installEventFilter(self)

    def _geom(self):
        w = max(380, int(self.stage.width() * 0.40))
        return self.stage.width() - w, w

    def eventFilter(self, obj, ev):
        if obj is self.stage and ev.type() == QEvent.Type.Resize and self.isVisible():
            x, w = self._geom()
            self.setGeometry(x, 0, w, self.stage.height())
        return False

    def open_drawer(self):
        x, w = self._geom()
        self.setGeometry(self.stage.width(), 0, w, self.stage.height())
        self.show()
        self.raise_()
        self.anim.stop()
        self.anim.setStartValue(QPoint(self.stage.width(), 0))
        self.anim.setEndValue(QPoint(x, 0))
        self.anim.start()

    def close_drawer(self):
        self.hide()

    def toggle(self):
        self.close_drawer() if self.isVisible() else self.open_drawer()

    def update_results(self, results, has_ecg, subtitle=""):
        self.sub.setText(subtitle)
        fill(self.t_cmp, scoring.COMPARISON, results)
        if has_ecg:
            fill(self.t_err, scoring.ERRORS, [r for r in results if r.vs_true is not None])
            self.err_stack.setCurrentIndex(0)
        else:
            self.err_stack.setCurrentIndex(1)


__all__ = ["MetricsDrawer", "theme"]
