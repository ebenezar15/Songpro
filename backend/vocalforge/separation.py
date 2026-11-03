"""Optional vocal separation (Demucs, MIT). Runs only if installed and consented."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from .engines.base import EngineUnavailable
from .engines.neural import consent_given


def separation_status():
    if importlib.util.find_spec("demucs") is None:
        return False, "Demucs is not installed (pip install -r requirements-ml.txt). ~320 MB of MIT-licensed weights."
    if not consent_given("demucs"):
        return False, "Demucs weights (~320 MB, MIT) need your consent before the first download."
    return True, "ready"


def separate_vocals(path: Path, out_dir: Path) -> Path:
    ok, msg = separation_status()
    if not ok:
        raise EngineUnavailable(msg)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", str(out_dir), str(path)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise EngineUnavailable("Demucs failed: " + proc.stderr[-600:])
    hits = list(out_dir.rglob("vocals.wav"))
    if not hits:
        raise EngineUnavailable("Demucs produced no vocals.wav")
    return hits[0]
