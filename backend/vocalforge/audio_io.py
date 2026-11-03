"""Audio import helpers. Everything stays local; ffmpeg is used only as a local decoder."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SUPPORTED_EXT = {".wav", ".aif", ".aiff", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}


class AudioImportError(RuntimeError):
    pass


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Return (samples[frames, channels] float32, sample_rate). Never silently fails."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXT:
        raise AudioImportError(f"Unsupported file type '{path.suffix}'. Use WAV, AIFF, MP3 or M4A.")
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return data, int(sr)
    except Exception:
        pass
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AudioImportError(f"Could not decode {path.name}: libsndfile failed and ffmpeg is not installed.")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "decoded.wav"
        proc = subprocess.run(
            [ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-i", str(path), "-c:a", "pcm_f32le", str(out)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not out.exists():
            raise AudioImportError(f"ffmpeg could not decode {path.name}: {proc.stderr.strip()[:300]}")
        data, sr = sf.read(str(out), dtype="float32", always_2d=True)
    return data, int(sr)


def to_mono(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=1) if data.ndim == 2 else data


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = np.gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g, axis=0).astype(np.float32)


def write_preview_wav(path: str | Path, x: np.ndarray, sr: int) -> None:
    """Float WAV for in-app playback (not the export path)."""
    sf.write(str(path), np.asarray(x, dtype=np.float32), sr, subtype="FLOAT")
