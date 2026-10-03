"""
Dark theme stylesheet for MagPy.

Ported verbatim from the legacy app's honed VS Code-style dark theme; applied
once on the main window. Kept as a module-level constant so the shell stays
readable and the look has a single home.
"""

DARK_STYLESHEET = """
QMainWindow {
    background-color: #1e1e1e;
}
QWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-family: 'Segoe UI', 'SF Pro Display', system-ui, sans-serif;
    font-size: 13px;
}
QMenuBar {
    background-color: #252526;
    border-bottom: 1px solid #3c3c3c;
    padding: 4px;
}
QMenuBar::item {
    padding: 6px 12px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #094771;
}
QMenu {
    background-color: #252526;
    border: 1px solid #3c3c3c;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item {
    padding: 8px 32px 8px 16px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #094771;
}
QMenu::separator {
    height: 1px;
    background-color: #3c3c3c;
    margin: 4px 8px;
}
QToolBar {
    background-color: #252526;
    border: none;
    spacing: 4px;
    padding: 4px;
}
QToolButton {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 6px;
}
QToolButton:hover {
    background-color: #3c3c3c;
    border-color: #4c4c4c;
}
QToolButton:pressed {
    background-color: #094771;
}
QStatusBar {
    background-color: #007acc;
    color: white;
}
QDockWidget::title {
    background-color: #252526;
    padding: 8px;
    border-bottom: 1px solid #3c3c3c;
}
QSplitter::handle {
    background-color: #3c3c3c;
}
QSplitter::handle:horizontal {
    width: 2px;
}
QSplitter::handle:vertical {
    height: 2px;
}
QScrollBar:vertical {
    background-color: #1e1e1e;
    width: 12px;
    border-radius: 6px;
}
QScrollBar::handle:vertical {
    background-color: #5a5a5a;
    border-radius: 6px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background-color: #787878;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background-color: #1e1e1e;
    height: 12px;
    border-radius: 6px;
}
QScrollBar::handle:horizontal {
    background-color: #5a5a5a;
    border-radius: 6px;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #787878;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
QTableWidget {
    background-color: #1e1e1e;
    gridline-color: #3c3c3c;
    border: none;
}
QTableWidget::item {
    padding: 4px 8px;
}
QTableWidget::item:selected {
    background-color: #094771;
}
QHeaderView::section {
    background-color: #252526;
    padding: 8px;
    border: none;
    border-right: 1px solid #3c3c3c;
    border-bottom: 1px solid #3c3c3c;
}
QPushButton {
    background-color: #0e639c;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    color: white;
}
QPushButton:hover {
    background-color: #1177bb;
}
QPushButton:pressed {
    background-color: #094771;
}
QPushButton:disabled {
    background-color: #3c3c3c;
    color: #808080;
}
QSlider::groove:horizontal {
    height: 4px;
    background-color: #3c3c3c;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background-color: #0e639c;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background-color: #1177bb;
}
QComboBox {
    background-color: #3c3c3c;
    border: 1px solid #5a5a5a;
    border-radius: 4px;
    padding: 6px 12px;
}
QComboBox:hover {
    border-color: #0e639c;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #3c3c3c;
    border: 1px solid #5a5a5a;
    border-radius: 4px;
    padding: 6px 8px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #0e639c;
}
"""
