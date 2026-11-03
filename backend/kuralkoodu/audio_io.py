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
    # macOS ships afconvert (Core Audio): it decodes MP3, M4A/AAC, ALAC and AIFF with no extra install.
    # ffmpeg (if present) covers everything else, e.g. WebM/Opus.
    decoders = []
    if shutil.which("afconvert"):
        decoders.append(("afconvert", lambda src, dst: [shutil.which("afconvert"), "-f", "WAVE", "-d", "LEF32", str(src), str(dst)]))
    if shutil.which("ffmpeg"):
        decoders.append(("ffmpeg", lambda src, dst: [shutil.which("ffmpeg"), "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-c:a", "pcm_f32le", str(dst)]))
    if not decoders:
        raise AudioImportError(f"Could not decode {path.name}. Save it as WAV or AIFF, or install ffmpeg for this format.")
    errors = []
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "decoded.wav"
        for name, cmd in decoders:
            proc = subprocess.run(cmd(path, out), capture_output=True, text=True)
            if proc.returncode == 0 and out.exists():
                data, sr = sf.read(str(out), dtype="float32", always_2d=True)
                break
            errors.append(f"{name}: {proc.stderr.strip()[:200]}")
        else:
            raise AudioImportError(f"Could not decode {path.name} ({'; '.join(errors)}). Save it as WAV or AIFF.")
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
