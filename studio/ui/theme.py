"""Colours and the Qt style sheet (one light theme; Fusion style so it looks the same on every Windows)."""

INK = "#111111"          # true ECG, main text
BLUE = "#2f6fd1"         # template baseline, accent
GREY = "#555555"         # other reconstructions
PPG = "#b5485d"          # the user's PPG
MUTED = "#6b7280"
LINE = "#e3e6eb"
PANEL = "#ffffff"
BG = "#f4f5f7"
RPEAK = "#c9ced6"
GREEN, AMBER, RED, HOLLOW = "#22a06b", "#d99a00", "#d64545", "#b8bec8"
ORANGE = "#e8740c"

FONT = "Segoe UI"

QSS = f"""
* {{ font-family: "{FONT}"; font-size: 10pt; color: {INK}; }}
QMainWindow, QWidget#central {{ background: {BG}; }}
QFrame#panel {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 10px; }}
QLabel#panelTitle {{ font-size: 11pt; font-weight: 600; }}
QLabel#muted, QLabel[muted="true"] {{ color: {MUTED}; }}
QLabel#small {{ color: {MUTED}; font-size: 9pt; }}

QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; padding: 7px 14px; margin-right: 2px; border: none;
               border-bottom: 2px solid transparent; color: {MUTED}; }}
QTabBar::tab:selected {{ color: {INK}; border-bottom: 2px solid {BLUE}; font-weight: 600; }}
QTabBar::tab:hover:!selected {{ color: {INK}; }}

QPushButton {{ background: #ffffff; border: 1px solid #cfd4dc; border-radius: 7px; padding: 6px 14px; }}
QPushButton:hover {{ border-color: {BLUE}; }}
QPushButton:disabled {{ color: #a0a6b0; border-color: {LINE}; background: #fafbfc; }}
QPushButton#primary {{ background: {BLUE}; color: white; border: 1px solid {BLUE}; font-weight: 600; }}
QPushButton#primary:hover {{ background: #2860b8; }}
QPushButton#primary:disabled {{ background: #a9c1ea; border-color: #a9c1ea; color: white; }}
QPushButton#danger {{ background: {RED}; color: white; border: 1px solid {RED}; font-weight: 600; }}
QPushButton#link {{ border: none; background: transparent; color: {BLUE}; padding: 0; text-decoration: underline; }}

QToolButton {{ border: 1px solid transparent; border-radius: 7px; padding: 4px; background: transparent; }}
QToolButton:hover {{ background: #eef2f8; border-color: #d6deea; }}
QToolButton:disabled {{ color: #b0b5bd; }}
QToolButton#play {{ color: {BLUE}; font-size: 11pt; }}
QToolButton#strip {{ font-size: 15pt; padding: 6px; }}
QToolButton#close {{ color: {MUTED}; font-size: 11pt; }}

QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{ background: white; border: 1px solid #cfd4dc; border-radius: 6px;
                                                  padding: 3px 6px; }}
QSpinBox:disabled, QDoubleSpinBox:disabled {{ color: #a0a6b0; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}

QFrame#dropZone {{ border: 2px dashed #b9c3d3; border-radius: 12px; background: #fbfcfe; }}
QFrame#dropZone[hover="true"] {{ border-color: {BLUE}; background: #f0f5ff; }}
QFrame#fileCard {{ border: 1px solid {LINE}; border-radius: 10px; background: #fbfcfe; }}
QFrame#modelRow {{ border-bottom: 1px solid {LINE}; background: transparent; }}
QFrame#modelRow[off="true"] QLabel {{ color: #a0a6b0; }}
QLabel#tag {{ color: {MUTED}; background: #eef1f5; border-radius: 8px; padding: 1px 8px; font-size: 8.5pt; }}

QProgressBar {{ border: none; background: #eef1f5; border-radius: 3px; max-height: 6px; }}
QProgressBar::chunk {{ background: {BLUE}; border-radius: 3px; }}

QFrame#drawer {{ background: white; border: 1px solid {LINE}; border-top-left-radius: 10px;
                 border-bottom-left-radius: 10px; }}
QTableWidget {{ border: none; gridline-color: {LINE}; selection-background-color: #e6eefb;
                selection-color: {INK}; }}
QHeaderView::section {{ background: #f7f8fa; border: none; border-bottom: 1px solid {LINE};
                        border-right: 1px solid {LINE}; padding: 5px 8px; font-weight: 600; }}

QFrame#toast {{ background: #1f2430; border-radius: 10px; }}
QFrame#toast QLabel {{ color: #f2f4f8; }}
QFrame#toast QLabel#toastTitle {{ font-weight: 600; }}
QFrame#toast QToolButton {{ color: #c8ccd4; }}

QDialog#popup {{ background: white; }}
QLabel#popupTitle {{ font-size: 15pt; font-weight: 600; }}
QLabel#popupText {{ font-size: 11pt; }}

QStatusBar {{ background: #eceef2; border-top: 1px solid {LINE}; }}
QStatusBar QLabel {{ color: {MUTED}; padding: 0 8px; }}
QMenuBar {{ background: {BG}; }}
QSplitter::handle {{ background: transparent; }}
QToolTip {{ background: #1f2430; color: #f2f4f8; border: none; padding: 5px 8px; border-radius: 5px; }}
"""
