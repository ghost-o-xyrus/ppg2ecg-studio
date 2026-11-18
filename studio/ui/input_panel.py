"""Top left: the input. Two tabs (PPG signal | True ECG), each a drop zone that turns into a file card."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QLabel, QStackedWidget, QTabWidget, QToolButton, QVBoxLayout, QWidget

from .widgets import DropZone, FileCard, Spinner


class InputTab(QWidget):
    fileChosen = Signal(str, str)      # path, kind
    removed = Signal(str)              # kind

    def __init__(self, kind, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.stack = QStackedWidget()
        self.drop = DropZone(kind)
        self.card = FileCard(kind)
        self.stack.addWidget(self.drop)
        self.stack.addWidget(self.card)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.addWidget(self.stack)
        self.drop.fileChosen.connect(lambda f: self.fileChosen.emit(f, self.kind))
        self.card.fileChosen.connect(lambda f: self.fileChosen.emit(f, self.kind))
        self.card.removed.connect(lambda: self.removed.emit(self.kind))

    def show_prepared(self, prep):
        if prep is None:
            self.stack.setCurrentWidget(self.drop)
        else:
            self.card.show_prepared(prep)
            self.stack.setCurrentWidget(self.card)


class InputPanel(QFrame):
    fileChosen = Signal(str, str)
    removed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 14)
        title = QLabel("Input")
        title.setObjectName("panelTitle")
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.ppg = InputTab("ppg")
        self.ecg = InputTab("ecg")
        self.tabs.addTab(self.ppg, "PPG signal")
        self.tabs.addTab(self.ecg, "True ECG (optional)")
        self.large = QToolButton(text="➕")
        self.large.setToolTip("Allow large files (up to 2 h): large-file mode arrives in a later version")
        self.large.setEnabled(False)
        self.tabs.setCornerWidget(self.large, Qt.TopRightCorner)
        lay.addWidget(title)
        lay.addWidget(self.tabs, 1)
        for t in (self.ppg, self.ecg):
            t.fileChosen.connect(self.fileChosen.emit)
            t.removed.connect(self.removed.emit)
        self.spinner = Spinner(self)

    def tab(self, kind) -> InputTab:
        return self.ppg if kind == "ppg" else self.ecg

    def show_prepared(self, kind, prep):
        self.tab(kind).show_prepared(prep)
        name = "PPG signal" if kind == "ppg" else "True ECG (optional)"
        if kind == "ecg" and prep is not None:
            name = "True ECG ✓"
        self.tabs.setTabText(0 if kind == "ppg" else 1, name)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.spinner.isVisible():
            self.spinner.setGeometry(self.rect())
