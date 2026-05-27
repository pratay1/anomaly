# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for anomaly.exe training GUI."""

import sys
from pathlib import Path

block_cipher = None
root = Path(SPECPATH)

a = Analysis(
    [str(root / "python" / "az" / "gui" / "app.py")],
    pathex=[str(root / "python")],
    binaries=[],
    datas=[],
    hiddenimports=[
        "az._az_core",
        "az.brain",
        "az.checkpoint",
        "az.config",
        "az.gui.app",
        "az.gui.main_window",
        "az.gui.multi_game_grid",
        "az.gui.solo_game_dialog",
        "az.gui.mini_board_view",
        "az.training.orchestrator",
        "az.training.central_learner",
        "az.training.selfplay_worker",
        "az.training.inference_server",
        "PyQt6.QtSvg",
        "pyqtgraph",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # App uses PyQt6; PyQt5 in site-packages triggers PyInstaller abort.
        "PyQt5",
        "PyQt5.QtCore",
        "PyQt5.QtGui",
        "PyQt5.QtWidgets",
        "PyQt5.QtSvg",
        "PyQt5.sip",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Bundle compiled C++ extension if present (editable install copy)
_az_core = root / "python" / "az" / "_az_core.pyd"
if not _az_core.exists():
    _az_core = root / "python" / "az" / "_az_core.so"
if _az_core.exists():
    a.binaries += [(_az_core.name, str(_az_core), "BINARY")]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="anomaly",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
