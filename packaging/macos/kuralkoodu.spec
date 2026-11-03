# PyInstaller spec for Kuralkoodu.app  (run from the repo root:  pyinstaller packaging/macos/kuralkoodu.spec)
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parents[1]
VERSION = (ROOT / "VERSION").read_text().strip()
is_mac = sys.platform == "darwin"

hidden = (collect_submodules("kuralkoodu") + collect_submodules("uvicorn") + ["multipart", "python_multipart"]
          # SciPy loads its array-API shims lazily; PyInstaller cannot see them
          + collect_submodules("scipy._external") + collect_submodules("scipy._lib"))
if is_mac:
    hidden += collect_submodules("webview")

a = Analysis(
    [str(ROOT / "packaging" / "macos" / "launcher.py")],
    pathex=[str(ROOT / "backend")],
    datas=[(str(ROOT / "frontend" / "dist"), "ui")] + collect_data_files("webview") if is_mac else [(str(ROOT / "frontend" / "dist"), "ui")],
    hiddenimports=hidden,
    excludes=["tkinter", "matplotlib", "pytest", "IPython", "torch", "demucs", "PyQt5", "PyQt6", "PySide2", "PySide6"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Kuralkoodu", console=not is_mac,
          target_arch=None, codesign_identity=None, entitlements_file=None)
coll = COLLECT(exe, a.binaries, a.datas, name="Kuralkoodu")

if is_mac:
    icon = ROOT / "packaging" / "macos" / "build" / "Kuralkoodu.icns"
    app = BUNDLE(
        coll,
        name="Kuralkoodu.app",
        icon=str(icon) if icon.exists() else None,
        bundle_identifier="app.kuralkoodu.studio",
        version=VERSION,
        info_plist={
            "CFBundleDisplayName": "Kuralkoodu",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "LSMinimumSystemVersion": "11.0",
            "NSHighResolutionCapable": True,
            "NSMicrophoneUsageDescription": "Kuralkoodu records your humming or singing to find the melody. Audio stays on this Mac.",
            "LSApplicationCategoryType": "public.app-category.music",
        },
    )
