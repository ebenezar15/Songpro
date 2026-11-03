"""FormantSingerEngine: a local, deterministic, source-filter singing synthesiser.

It needs no model download and runs on CPU anywhere. It speaks every phoneme in the
Tamil inventory (retroflex ʈ ɖ ɳ ɭ ɻ, dental t̪ d̪ n̪, tap/trill, unaspirated stops), so
pronunciation rules are audible, but the timbre is clearly synthetic. Treat it as a
pronunciation-accurate guide / fallback; neural engines are for release quality.

Signal path per 5 ms frame:
  harmonic source (additive, glottal tilt, jitter/shimmer)
  x cascade formant envelope (F1..F5, scaled by vocal-tract length) + singer's formant
  + aspiration noise shaped by the same envelope (breathiness, pitch-synchronous)
  + frication / burst noise shaped per place of articulation
  -> grit (period-doubling + soft saturation)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import istft, stft

from ..phonemes import lookup
from .base import Cancelled, VocalEngine

SR = 48000
HOP = SR // 200  # 240 samples = 5 ms (matches planner FPS)
HARM_CUTOFF = 11000.0


def _paint_params(plan, voice, timbre_off):
    n = plan["n_frames"]
    F = np.tile(np.array([700, 1250, 2550, 3400, 4000], float), (n, 1))
    B = np.tile(np.array([80, 100, 140, 220, 280], float), (n, 1))
    vg = np.zeros(n)  # voiced gain
    ng = np.zeros(n)  # frication noise gain
    nc = np.full(n, 3000.0)
    nb = np.full(n, 2000.0)
    lp = np.full(n, 20000.0)
    nasal = np.zeros(n)
    trill = np.zeros(n)
    last_F = F[0].copy()
    segs = sorted(plan["phones"], key=lambda s: s["start"])
    for i, sg in enumerate(segs):
        a = max(0, int(round(sg["start"] * 200)))
        b = min(n, max(a + 1, int(round(sg["end"] * 200))))
        p = lookup(sg["ph"])
        kind = p["kind"]
        Fp = p.get("F")
        if Fp is None:  # /h/: take next segment's formants
            nxt = next((lookup(s["ph"]).get("F") for s in segs[i + 1:] if lookup(s["ph"]).get("F")), last_F)
            Fp = nxt
        F[a:b] = Fp
        B[a:b] = p["B"]
        last_F = np.array(Fp, float)
        L = b - a
        if kind == "vowel":
            vg[a:b] = 1.0
        elif kind in ("stop", "affricate"):
            clos = int(L * (0.6 if kind == "stop" else 0.35))
            vg[a:a + clos] = p["voiced"] * 0.35
            lp[a:a + clos] = 450.0
            rel = slice(a + clos, b)
            vg[rel] = p["voiced"] * 0.6
            if kind == "stop":
                burst = slice(a + clos, min(b, a + clos + 3))
                ng[burst] = p["noise"] * 1.3
            else:
                ng[rel] = p["noise"] * 0.9
            nc[a:b], nb[a:b] = p["ncenter"], p["nbw"]
        elif kind == "fric":
            vg[a:b] = p["voiced"] * 0.7
            ng[a:b] = p["noise"]
            nc[a:b], nb[a:b] = p["ncenter"], p["nbw"]
        elif kind == "nasal":
            vg[a:b] = p["gain"]
            nasal[a:b] = 1.0
            lp[a:b] = 2600.0
        elif kind == "tap":
            vg[a:b] = 1.0
            mid = a + L // 2
            vg[max(a, mid - 1):mid + 1] = 0.25
        elif kind == "trill":
            vg[a:b] = p["gain"]
            trill[a:b] = 1.0
        else:  # approx
            vg[a:b] = p["gain"]
            ng[a:b] = p["noise"]
            nc[a:b], nb[a:b] = p.get("ncenter", 3000), p.get("nbw", 2000)
    # coarticulation: smooth formant tracks (~25 ms) and gains (~10 ms)
    F = uniform_filter1d(F, 5, axis=0)
    B = uniform_filter1d(B, 3, axis=0)
    vg = uniform_filter1d(vg, 2)
    nasal = uniform_filter1d(nasal, 3)
    lp = np.exp(uniform_filter1d(np.log(lp), 2))
    scale = voice["formant_scale"] * (1 + timbre_off.get("formant", 0.0))
    F = F * scale
    return dict(F=F, B=B * (0.9 + 0.1 * scale), vg=vg, ng=ng, nc=nc * np.sqrt(scale), nb=nb, lp=lp, nasal=nasal, trill=trill)


def _envelope(freqs, F, B, nasal, lp, bright, scale):
    """Parallel formant magnitude at `freqs` [frames, K] (peak-normalised resonances,
    amplitudes falling ~6 dB/oct with formant frequency, as in Klatt's parallel branch)."""
    H = np.zeros_like(freqs)
    gains = (1.0, 0.75, 0.42, 0.28, 0.18)
    for i in range(F.shape[1]):
        Fi = F[:, i:i + 1]
        Bi = B[:, i:i + 1]
        R = 1.0 / np.sqrt(1.0 + ((freqs ** 2 - Fi ** 2) / (Bi * freqs + 1e-9)) ** 2)
        H += gains[i] * (F[:, :1] / Fi) ** 0.5 * R
    H += 0.012  # inter-formant floor
    # singer's formant cluster (brightness)
    sf = 2900.0 * scale
    H *= 1 + (0.9 * bright) * np.exp(-0.5 * ((freqs - sf) / 450.0) ** 2)
    # nasal: anti-resonance near 1 kHz
    nz = nasal[:, None]
    H *= 1 - 0.85 * nz * np.exp(-0.5 * ((freqs - 1100 * scale) / 250.0) ** 2)
    H *= 1 / np.sqrt(1 + (freqs / lp[:, None]) ** 6)
    return H


class FormantSingerEngine(VocalEngine):
    id = "formant"
    name = "VocalForge Formant Singer (local)"
    license_id = "vocalforge"
    languages = ("ta", "ta-Latn", "en")
    native_sr = SR
    version = "formant-1.0"

    def render(self, plan, seed, progress=lambda f, m: None, cancel=lambda: False):
        voice = plan["voice"]
        timbre = plan.get("timbre", {})
        toff = plan.get("timbre_offset", {})
        seed = int(seed) + int(toff.get("seed_offset", 0))
        rng = np.random.default_rng([seed, 1])
        breath = float(np.clip(voice.get("breathiness", 0.15) + 0.6 * timbre.get("breathiness", 0.0), 0, 0.95))
        grit = float(np.clip(timbre.get("grit", 0.0), 0, 1))
        bright = float(voice.get("brightness", 0.6))
        scale = voice["formant_scale"]

        nfr = plan["n_frames"]
        f0 = np.maximum(np.asarray(plan["f0_hz"], float), 40.0)
        amp = np.asarray(plan["amp"], float)
        P = _paint_params(plan, voice, toff)
        progress(0.1, "Formant tracks built")

        # shimmer
        amp_s = amp * (1 + 0.04 * uniform_filter1d(rng.normal(0, 1, nfr), 3))
        # trill: ~26 Hz amplitude modulation
        tr_t = np.arange(nfr) / 200.0
        trill_mod = 1 - P["trill"] * 0.6 * (0.5 + 0.5 * np.cos(2 * np.pi * 26 * tr_t))
        voiced_amp = amp_s * P["vg"] * trill_mod * (1 - 0.55 * breath)

        K = int(HARM_CUTOFF / max(60.0, f0.min())) + 1
        K = min(K, 200)
        k = np.arange(1, K + 1)[None, :]
        freqs = f0[:, None] * k
        tilt = 1.0 / k ** (0.55 + 0.45 * (1 - bright) + 0.6 * breath)
        H = _envelope(freqs, P["F"], P["B"], P["nasal"], P["lp"], bright, scale)
        A = H * tilt
        A[freqs > HARM_CUTOFF] = 0
        # frame-wise level normalisation (partial: keeps natural vowel differences)
        norm = np.sqrt((A ** 2).sum(axis=1)) + 1e-9
        A = A / norm[:, None] ** 0.85
        A *= voiced_amp[:, None]
        progress(0.25, "Harmonic amplitudes ready")

        # sample-level synthesis in chunks
        n = nfr * HOP
        fpos = np.arange(n) / HOP
        f0s = np.interp(fpos, np.arange(nfr), f0)
        phase = 2 * np.pi * np.cumsum(f0s) / SR
        y = np.zeros(n, np.float64)
        chunk = 9600
        for c0 in range(0, n, chunk):
            if cancel():
                raise Cancelled()
            c1 = min(n, c0 + chunk)
            fp = fpos[c0:c1]
            i0 = np.minimum(fp.astype(int), nfr - 2)
            w = (fp - i0)[:, None]
            Ac = A[i0] * (1 - w) + A[i0 + 1] * w
            active = np.nonzero(Ac.max(axis=0) > 1e-6)[0]
            if active.size:
                ph = phase[c0:c1, None] * (active[None, :] + 1)
                y[c0:c1] = np.einsum("ij,ij->i", Ac[:, active], np.sin(ph))
            progress(0.25 + 0.5 * c1 / n, "Synthesising harmonics")

        # grit: period doubling + saturation
        if grit > 0:
            sub = 1 - 0.35 * grit * (0.5 + 0.5 * np.cos(phase / 2))
            y = y * sub
            drive = 1 + 5 * grit
            y = np.tanh(drive * y / (np.max(np.abs(y)) + 1e-9)) / np.tanh(drive) * np.max(np.abs(y))

        # noise streams (aspiration + frication), shaped in the STFT domain
        nper = 1024
        noise = rng.normal(0, 1, n + nper)
        f_axis, t_axis, Z = stft(noise, fs=SR, nperseg=nper, noverlap=nper - HOP)
        nF = Z.shape[1]
        fi = np.clip((t_axis * 200).astype(int), 0, nfr - 1)
        fr = f_axis[None, :]
        Hn = _envelope(np.broadcast_to(fr, (nF, len(f_axis))).copy(), P["F"][fi], P["B"][fi] * 1.6,
                       P["nasal"][fi], np.full(nF, 20000.0), bright, scale)
        Hn = Hn / (Hn.max(axis=1, keepdims=True) + 1e-9)
        asp = Hn * (breath * 0.6 * amp_s[fi] * np.clip(P["vg"][fi], 0, 1))[:, None]
        fric_shape = np.exp(-0.5 * ((fr - P["nc"][fi][:, None]) / (P["nb"][fi][:, None] / 2)) ** 2)
        fric = fric_shape * (0.5 * P["ng"][fi] * np.maximum(amp_s[fi], 0.3 * (P["ng"][fi] > 0)))[:, None]
        hf_roll = 1 / np.sqrt(1 + (fr / 9000.0) ** 4)
        Zs = Z * ((asp + fric) * hf_roll).T * 0.045
        _, yn = istft(Zs, fs=SR, nperseg=nper, noverlap=nper - HOP)
        yn = np.pad(yn, (0, max(0, n - len(yn))))[:n]
        # aspiration is pitch-synchronous in real voices
        yn *= 0.75 + 0.25 * np.cos(phase)
        out = 0.22 * y + yn
        # gentle DC / rumble removal
        out = out - uniform_filter1d(out, 1920)
        progress(1.0, "Formant render done")
        return out.astype(np.float32), SR
