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
    license_id = "kuralkoodu"
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
        breath = float(plan.get("timbre", {}).get("breathiness", 0))
        # long songs: 20 s blocks with 0.5 s raised-cosine crossfades
        sr2 = self.native_sr
        B, O = 20 * sr2, sr2 // 2
        out = np.zeros(len(x))
        wsum = np.zeros(len(x))
        starts = list(range(0, max(1, len(x)), B))
        for bi, s0 in enumerate(starts):
            if cancel():
                raise Cancelled()
            a0, a1 = max(0, s0 - O), min(len(x), s0 + B + O)
            seg = self._convert(x[a0:a1], semis, alpha, breath, seed, bi)
            w = np.ones(a1 - a0)
            if a0 > 0:
                w[:2 * O] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, 2 * O))
            if a1 < len(x):
                w[-2 * O:] = np.minimum(w[-2 * O:], 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, 2 * O)))
            out[a0:a1] += seg * w
            wsum[a0:a1] += w
            progress((bi + 1) / len(starts), f"Converted {min(len(x), s0 + B) / sr2:.0f} s")
        out /= np.maximum(wsum, 1e-6)
        return out.astype(np.float32), self.native_sr

    def _convert(self, x, semis, alpha, breath, seed, block):
        r = 2 ** (semis / 12)
        y = x
        if abs(semis) > 0.01:
            y = phase_vocoder_stretch(x, r)
            y = resample_poly(y, 1000, int(round(1000 * r)))
        y = np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)]
        _, _, X = stft(x, nperseg=N_FFT, noverlap=N_FFT - HOP)
        _, _, Y = stft(y, nperseg=N_FFT, noverlap=N_FFT - HOP)
        m = min(X.shape[1], Y.shape[1])
        X, Y = X[:, :m], Y[:, :m]
        eX, eY = _envelope(np.abs(X)), _envelope(np.abs(Y))
        bins = np.arange(X.shape[0])
        src = np.clip(bins / alpha, 0, bins[-1])
        i0 = np.floor(src).astype(int)
        i1 = np.minimum(i0 + 1, bins[-1])
        fw = (src - i0)[:, None]
        target = eX[i0] * (1 - fw) + eX[i1] * fw
        Yc = Y * np.clip(target / (eY + 1e-9), 0, 8)
        if breath > 0:
            rng = np.random.default_rng([int(seed), 5, block])
            noise = (rng.normal(size=Yc.shape) + 1j * rng.normal(size=Yc.shape)) * np.abs(Yc) * 0.35 * breath
            Yc = Yc + noise * (bins / bins[-1])[:, None] ** 0.5
        _, out = istft(Yc, nperseg=N_FFT, noverlap=N_FFT - HOP)
        return np.pad(out, (0, max(0, len(x) - len(out))))[: len(x)]
