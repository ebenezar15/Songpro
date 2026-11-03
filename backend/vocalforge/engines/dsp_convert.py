"""DspConvertEngine: keeps a sung vocal's words, timing and pronunciation (any language,
including Tamil) and changes only voice character: formant (vocal-tract) scaling, transposition,
breathiness. Local, deterministic, no model download. Quality: clean for moderate changes
(+-4 semitones, formant 0.9-1.2); extreme gender flips sound processed."""
from __future__ import annotations

import numpy as np
from scipy.signal import istft, resample_poly, stft

from .base import Cancelled, EngineUnavailable, VocalEngine

N_FFT = 2048
HOP = 256


def _envelope(mag, lifter=40):
    c = np.fft.irfft(np.log(mag + 1e-9), axis=0)
    c[lifter:-lifter] = 0
    return np.exp(np.fft.rfft(c, axis=0).real)


def phase_vocoder_stretch(x, rate):
    """Time-stretch by `rate` (>1 = longer) without changing pitch."""
    _, _, Z = stft(x, nperseg=N_FFT, noverlap=N_FFT - HOP)
    n_bins, n_fr = Z.shape
    steps = np.arange(0, n_fr - 1, 1 / rate)
    omega = 2 * np.pi * HOP * np.arange(n_bins) / N_FFT
    phase = np.angle(Z[:, 0])
    out = np.zeros((n_bins, len(steps)), complex)
    for i, s in enumerate(steps):
        k = int(s)
        frac = s - k
        a, b = Z[:, k], Z[:, min(k + 1, n_fr - 1)]
        mag = (1 - frac) * np.abs(a) + frac * np.abs(b)
        out[:, i] = mag * np.exp(1j * phase)
        dphi = np.angle(b) - np.angle(a) - omega
        dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))
        phase = phase + omega + dphi
    _, y = istft(out, nperseg=N_FFT, noverlap=N_FFT - HOP)
    return y


class DspConvertEngine(VocalEngine):
    id = "dsp-convert"
    name = "Voice Character (DSP, keeps sung words)"
    license_id = "vocalforge"
    languages = ("any (keeps the sung words)",)
    input_types = ("sung",)
    native_sr = 48000
    version = "dsp-1.0"

    def render(self, plan, seed, progress=lambda f, m: None, cancel=lambda: False):
        src = plan.get("source_audio")
        if src is None:
            raise EngineUnavailable("Voice Character mode needs a sung vocal as input.")
        x, sr = src
        from ..audio_io import resample
        x = resample(np.asarray(x, np.float32), sr, self.native_sr).astype(np.float64)
        semis = float(plan.get("transpose", 0))
        alpha = float(plan["voice"]["formant_scale"]) / float(plan.get("source_formant_scale", 1.0))
        r = 2 ** (semis / 12)
        progress(0.1, "Pitch shifting")
        y = x
        if abs(semis) > 0.01:
            y = phase_vocoder_stretch(x, r)
            up, down = 1000, int(round(1000 * r))
            y = resample_poly(y, up, down)
        y = np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)]
        if cancel():
            raise Cancelled()
        progress(0.5, "Re-shaping formants")
        _, _, X = stft(x, nperseg=N_FFT, noverlap=N_FFT - HOP)
        _, _, Y = stft(y, nperseg=N_FFT, noverlap=N_FFT - HOP)
        m = min(X.shape[1], Y.shape[1])
        X, Y = X[:, :m], Y[:, :m]
        eX, eY = _envelope(np.abs(X)), _envelope(np.abs(Y))
        bins = np.arange(X.shape[0])
        target = np.stack([np.interp(bins / alpha, bins, eX[:, j]) for j in range(m)], axis=1)
        gain = np.clip(target / (eY + 1e-9), 0, 8)
        breath = float(plan.get("timbre", {}).get("breathiness", 0))
        Yc = Y * gain
        if breath > 0:
            rng = np.random.default_rng([int(seed), 5])
            noise = (rng.normal(size=Yc.shape) + 1j * rng.normal(size=Yc.shape)) * np.abs(Yc) * 0.35 * breath
            hf = (bins / bins[-1])[:, None] ** 0.5
            Yc = Yc + noise * hf
        _, out = istft(Yc, nperseg=N_FFT, noverlap=N_FFT - HOP)
        out = np.pad(out, (0, max(0, len(x) - len(out))))[: len(x)]
        progress(1.0, "Done")
        return out.astype(np.float32), self.native_sr
