"""Dark monochrome palette — premium understated UI."""

# Surfaces
BG_DEEPEST = "#0a0a0a"
BG_PRIMARY = "#1a1a1a"
BG_SECONDARY = "#1e1e1e"
BG_ELEVATED = "#252525"
BG_INTERACTIVE = "#2a2a2a"
BG_GUNMETAL = "#333333"
BORDER_SUBTLE = "#2a2a2a"
BORDER_MUTED = "#383838"

# Text
TEXT_PRIMARY = "#e5e5e5"
TEXT_SECONDARY = "#a0a0a0"
TEXT_DISABLED = "#606060"

# Accents (monochrome — no bright colors)
ACCENT_SOFT = "#8a8a8a"
ACCENT_MID = "#6a6a6a"
ACCENT_DIM = "#4a4a4a"
HEAT_LOW = "#181818"
HEAT_MID = "#3a3a3a"
HEAT_HIGH = "#7a7a7a"
THINKING_GLOW = "#505050"

# Chess board (refined monochrome checker — better contrast, warmer charcoals)
LIGHT_SQUARE = "#2f2f2f"
DARK_SQUARE = "#1c1c1c"
HIGHLIGHT = "#8a8a8a66"
MOVE_HINT = "#a0a0a066"
LAST_MOVE_HIGHLIGHT = "#bcbcbc35"
COORD_TEXT = "#5a5a5a"

# Metrics curve tints
METRIC_COLORS = {
    "policy_loss": "#9a9a9a",
    "value_loss": "#7a7a7a",
    "total_loss": "#b0b0b0",
    "lr": "#6a6a6a",
    "games_per_min": "#8a8a8a",
    "win_rate": "#a8a8a8",
}

DARK_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', 'Inter', system-ui, sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    background-color: {BG_SECONDARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
    margin-top: 10px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {TEXT_SECONDARY};
}}
QPushButton {{
    background-color: {BG_INTERACTIVE};
    border: 1px solid {BORDER_MUTED};
    border-radius: 6px;
    padding: 10px 20px;
    color: {TEXT_PRIMARY};
}}
QPushButton:hover {{
    background-color: {BG_GUNMETAL};
}}
QPushButton:pressed {{
    background-color: {BG_ELEVATED};
}}
QListWidget {{
    background-color: {BG_DEEPEST};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
    color: {TEXT_PRIMARY};
}}
QListWidget::item {{
    padding: 6px 8px;
    border-radius: 4px;
}}
QListWidget::item:selected {{
    background-color: {BG_GUNMETAL};
    color: {TEXT_PRIMARY};
}}
QListWidget::item:alternate {{
    background-color: {BG_ELEVATED};
}}
QGroupBox#metrics_panel, QGroupBox#games_panel {{
    background-color: {BG_SECONDARY};
}}
QStatusBar {{
    background-color: {BG_DEEPEST};
    color: {TEXT_SECONDARY};
    border-top: 1px solid {BORDER_SUBTLE};
}}
QLabel#title {{
    font-size: 22px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    letter-spacing: 0.8px;
}}
QLabel#subtitle {{
    font-size: 11px;
    color: {TEXT_DISABLED};
    letter-spacing: 1.2px;
    text-transform: uppercase;
}}
QLabel#mcts_status {{
    color: {TEXT_SECONDARY};
    font-size: 11px;
}}
QFrame#board_frame {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #131313, stop:1 {BG_DEEPEST});
    border: 1px solid {BORDER_MUTED};
    border-radius: 12px;
    padding: 8px;
}}
QGroupBox#mcts_panel {{
    padding-top: 14px;
    margin-top: 4px;
}}
QFrame#mcts_card {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
}}
QFrame#mcts_card:hover {{
    border-color: {BORDER_MUTED};
    background-color: {BG_GUNMETAL};
}}
QFrame#mcts_card_skeleton {{
    background-color: {BG_ELEVATED};
    border: 1px dashed {BORDER_SUBTLE};
    border-radius: 6px;
}}
QFrame#mcts_card_skeleton QLabel#mcts_move,
QFrame#mcts_card_skeleton QLabel#mcts_stat {{
    color: {TEXT_DISABLED};
}}
QLabel#mcts_move {{
    color: {TEXT_PRIMARY};
    font-size: 13px;
    font-weight: 600;
    font-family: 'Consolas', 'Cascadia Mono', monospace;
}}
QLabel#mcts_stat {{
    color: {TEXT_DISABLED};
    font-size: 9px;
}}
QSplitter::handle {{
    background-color: {BORDER_SUBTLE};
    width: 2px;
}}
"""
