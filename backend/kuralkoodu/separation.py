"""Vocal / music separation.

Two methods behind one function:

* "builtin"  (no download, CPU, deterministic). A classic signal-processing separator:
    1. Lead-melody tracking (harmonic-summation salience) and a harmonic comb mask that keeps only
       the singer's partials.
    2. REPET-SIM: accompaniment repeats, so each frame's background is the median of the most similar
       frames elsewhere in the song (searched within +-45 s, so 4+ minute songs stay fast). What does
       not repeat above 3.5 kHz is kept as consonants and breaths.
    3. Stereo: gentle preference for centre-panned, in-phase content.
    4. Vocal-band shaping (removes the bass / kick region).
   The vocal is mask-filtered; the instrumental is mix - vocal, so the two always sum to the original.
   Good enough to analyse a melody or as a guide; expect audible bleed on dense mixes.
* "demucs"   (Meta's htdemucs, MIT licence, ~80 MB weights, PyTorch; MPS on Apple Silicon). Much cleaner.
   Only runs when installed and after the user consents to the weight download.
"""
from __future__ import annotations

import importlib.util

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from .engines.base import Cancelled, EngineUnavailable
from .engines.neural import consent_given

METHODS = {
    "builtin": {"name": "Quick (built-in)", "license_id": "kuralkoodu", "download_mb": 0,
                "notes": "Runs immediately. Expect some music bleed in dense mixes."},
    "demucs": {"name": "Studio (Demucs htdemucs)", "license_id": "demucs", "download_mb": 80,
               "notes": "Cleaner separation. Needs PyTorch and a one-time weight download."},
}
N_FFT = 4096
HF_KEEP = 0.6  # how much non-repeating >3.5 kHz energy (consonants, breaths) is kept
HOP = 1024


def separation_status(method: str = "demucs"):
    if method == "builtin":
        return True, "ready"
    if importlib.util.find_spec("demucs") is None or importlib.util.find_spec("torch") is None:
        return False, "Demucs is not installed (pip install -r requirements-ml.txt). Weights ~80 MB, MIT licence."
    if not consent_given("demucs"):
        return False, "Demucs weights (~80 MB, MIT) need your consent before the first download."
    return True, "ready"


def separate(x: np.ndarray, sr: int, method="builtin", progress=lambda f, m: None, cancel=lambda: False, strength=1.0):
    """x: [frames, channels] float. Returns (vocals[frames, ch], instrumental[frames, ch], info)."""
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if method == "demucs":
        return _demucs(x, sr, progress, cancel)
    if method != "builtin":
        raise EngineUnavailable(f"Unknown separation method '{method}'")
    return _builtin(x, sr, progress, cancel, strength)


# ------------------------------------------------------------------ built-in
_WIN = np.hanning(N_FFT + 1)[:-1].astype(np.float32)  # periodic Hann


def _stft(x):
    """float32 STFT computed in blocks -> complex64 [bins, frames] (memory-lean for long songs)."""
    xp = np.pad(x.astype(np.float32), (N_FFT // 2, N_FFT))
    n_fr = 1 + len(x) // HOP
    Z = np.empty((N_FFT // 2 + 1, n_fr), np.complex64)
    for b0 in range(0, n_fr, 1024):
        b1 = min(n_fr, b0 + 1024)
        idx = np.arange(N_FFT)[None, :] + HOP * np.arange(b0, b1)[:, None]
        Z[:, b0:b1] = np.fft.rfft(xp[idx] * _WIN, axis=1).T
    return Z


def _istft(Z, n):
    """Weighted overlap-add inverse of _stft."""
    out = np.zeros(n + 2 * N_FFT, np.float32)
    wsum = np.zeros_like(out)
    w2 = _WIN * _WIN
    n_fr = Z.shape[1]
    for b0 in range(0, n_fr, 1024):
        b1 = min(n_fr, b0 + 1024)
        fr = (np.fft.irfft(Z[:, b0:b1].T, N_FFT, axis=1) * _WIN).astype(np.float32)
        for j, f in enumerate(fr):
            s = (b0 + j) * HOP
            out[s:s + N_FFT] += f
            wsum[s:s + N_FFT] += w2
    out = out[N_FFT // 2: N_FFT // 2 + n] / np.maximum(wsum[N_FFT // 2: N_FFT // 2 + n], 1e-6)
    return out

def _features(V, sr):
    """Log-compressed, band-averaged spectra for frame similarity (cheap, robust)."""
    hz = sr / N_FFT
    edges = np.unique(np.round(np.geomspace(60 / hz, min(12000, sr / 2 - 1) / hz, 65)).astype(int))
    bands = np.stack([V[lo:max(hi, lo + 1)].mean(axis=0) for lo, hi in zip(edges[:-1], edges[1:])])
    f = np.log1p(bands * 50).T  # frames x 64
    f -= f.mean(axis=1, keepdims=True)
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-9
    return f.astype(np.float32)


def repet_sim_background(V, feats, fps, k=24, min_gap_s=1.0, window_s=45.0, progress=None, cancel=None):
    n = V.shape[1]
    bg = np.empty_like(V)
    gap = int(min_gap_s * fps)
    win = int(window_s * fps)
    BLK = 256
    for b0 in range(0, n, BLK):
        if cancel and cancel():
            raise Cancelled()
        b1 = min(n, b0 + BLK)
        lo, hi = max(0, b0 - win), min(n, b1 + win)
        sim = feats[b0:b1] @ feats[lo:hi].T  # (blk, window)
        rows = np.arange(b0, b1)[:, None]
        cols = np.arange(lo, hi)[None, :]
        sim[np.abs(rows - cols) < gap] = -np.inf
        kk = min(k, sim.shape[1] - 1)
        if kk < 3:
            bg[:, b0:b1] = V[:, b0:b1]
            continue
        idx = np.argpartition(-sim, kk, axis=1)[:, :kk] + lo
        # median over the k most similar frames, per bin
        for i, r in enumerate(range(b0, b1)):
            bg[:, r] = np.median(V[:, idx[i]], axis=1)
        if progress:
            progress(0.15 + 0.5 * b1 / n, "Finding repeating music (REPET-SIM)")
    return np.minimum(bg, V)


def predominant_f0(W, sr, fmin_midi=40, fmax_midi=84):
    """Harmonic-summation salience over quarter-tone candidates. Returns f0 (Hz), salience ratio per frame."""
    hz = sr / N_FFT
    cands = np.arange(fmin_midi * 4, fmax_midi * 4 + 1) / 4
    f0s = 440 * 2 ** ((cands - 69) / 12)
    sal = np.zeros((len(cands), W.shape[1]), np.float32)
    LW = np.log1p(W * 20)
    for h in range(1, 9):
        idx = np.clip(np.round(f0s * h / hz).astype(int), 0, W.shape[0] - 1)
        sal += (0.85 ** (h - 1)) * LW[idx]
    best = median_filter(sal.argmax(axis=0), size=5)  # suppress octave flips
    ratio = sal.max(axis=0) / (np.median(sal, axis=0) + 1e-9)
    return f0s[best], ratio


def _builtin(x, sr, progress, cancel, strength):
    ch = x.shape[1]
    progress(0.02, "Analysing stereo image")
    specs = []
    for c in range(ch):
        specs.append(_stft(x[:, c]))
    if cancel():
        raise Cancelled()
    mid = specs[0] if ch == 1 else 0.5 * (specs[0] + specs[1])
    V = np.abs(mid).astype(np.float32)
    fps = sr / HOP
    hz = sr / N_FFT
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr).astype(np.float32)
    fc = freqs[:, None]
    band = (1 / (1 + (110 / (fc + 1)) ** 4) / (1 + (fc / 14000) ** 4)).astype(np.float32)

    # 1. lead melody: predominant pitch, then a harmonic comb that keeps only the singer's partials
    progress(0.1, "Tracking the lead melody")
    f0, ratio = predominant_f0(V * band, sr)
    comb = np.zeros_like(V)
    BLK = 2048
    for b0 in range(0, V.shape[1], BLK):
        if cancel():
            raise Cancelled()
        b1 = min(V.shape[1], b0 + BLK)
        fb = f0[None, b0:b1].astype(np.float32)
        cb = comb[:, b0:b1]
        for h in range(1, 60):
            fh = fb * h
            if float(fh.min()) > 9000:
                break
            np.maximum(cb, np.exp(-0.5 * ((fc - fh) / (hz * 1.2 + 0.015 * fh)) ** 2), out=cb)
        progress(0.1 + 0.4 * b1 / V.shape[1], "Following the singer's harmonics")
    # melody salience is reported, but not used as a gate: measured, it removed more vocal than music
    act = np.clip((ratio - np.percentile(ratio, 20)) / (np.percentile(ratio, 80) - np.percentile(ratio, 20) + 1e-9), 0, 1)

    # 2. consonants / breaths: non-repeating high-frequency energy (REPET-SIM background removal)
    progress(0.55, "Finding repeating music (REPET-SIM)")
    feats = _features(V, sr)
    bg = repet_sim_background(V, feats, fps, progress=lambda f, m: progress(0.55 + 0.25 * (f - 0.15) / 0.5, m), cancel=cancel)
    hf = (fc > 3500).astype(np.float32)
    progress(0.85, "Rebuilding vocal and instrumental")

    # 3. combine per frame block (stereo centre preference applied gently), masking the spectra in place
    for b0 in range(0, V.shape[1], BLK):
        b1 = min(V.shape[1], b0 + BLK)
        Vb, Bb = V[:, b0:b1], bg[:, b0:b1]
        d = np.maximum(Vb - Bb, 0)
        m_rep = d * d / (d * d + Bb * Bb + 1e-9)
        Mb = np.maximum(comb[:, b0:b1], HF_KEEP * m_rep * hf) * band
        if ch >= 2:
            Lb, Rb = specs[0][:, b0:b1], specs[1][:, b0:b1]
            aL, aR = np.abs(Lb), np.abs(Rb)
            coh = 2 * np.abs(Lb * np.conj(Rb)) / (aL * aL + aR * aR + 1e-9)
            pan = (aL - aR) / (aL + aR + 1e-9)
            Mb *= 0.5 + 0.5 * coh * np.exp(-(pan / 0.35) ** 2)
        Mb = np.clip(Mb, 0, 1) ** (1.0 / max(0.3, strength))
        for c in range(ch):
            specs[c][:, b0:b1] *= Mb.astype(np.float32)
    del comb, bg, V

    vocals = np.zeros_like(x)
    for c in range(ch):
        vocals[:, c] = _istft(specs[c], len(x))
        specs[c] = None
    instrumental = x - vocals  # exact complement: vocals + instrumental == mix
    progress(1.0, "Separation done")
    return vocals, instrumental, {"method": "builtin", "sample_rate": sr, "channels": ch,
                                  "vocal_activity": round(float(np.mean(act > 0.6)), 3)}


# -------------------------------------------------------------------- demucs
def _demucs(x, sr, progress, cancel):
    ok, msg = separation_status("demucs")
    if not ok:
        raise EngineUnavailable(msg)
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    from .audio_io import resample
    progress(0.05, "Loading htdemucs")
    model = get_model("htdemucs")
    model.eval()
    dev = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    msr = model.samplerate
    y = resample(x, sr, msr)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    wav = torch.from_numpy(y.T.copy())
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    progress(0.15, f"Separating on {dev} (long songs are split into overlapping segments)")
    if cancel():
        raise Cancelled()
    with torch.no_grad():
        out = apply_model(model, wav[None], device=dev, shifts=1, split=True, overlap=0.25, progress=False)[0]
    out = out * (ref.std() + 1e-8) + ref.mean()
    vi = model.sources.index("vocals")
    voc = out[vi].cpu().numpy().T
    voc = resample(voc.astype(np.float32), msr, sr)[: len(x)]
    voc = np.pad(voc, ((0, max(0, len(x) - len(voc))), (0, 0)))
    if x.shape[1] == 1:
        voc = voc.mean(axis=1, keepdims=True)
    progress(1.0, "Separation done")
    return voc, x - voc, {"method": "demucs", "model": "htdemucs", "device": dev, "sample_rate": sr, "channels": x.shape[1]}


def sdr(est, ref):
    """Signal-to-distortion ratio in dB (simple, scale-sensitive)."""
    est, ref = np.asarray(est, float).ravel(), np.asarray(ref, float).ravel()
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    return float(10 * np.log10((ref ** 2).sum() / (((ref - est) ** 2).sum() + 1e-12) + 1e-12))
