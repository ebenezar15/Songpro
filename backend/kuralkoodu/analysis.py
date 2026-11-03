"""Melody / vocal analysis: F0, notes, tempo, key, dynamics, vibrato, slides, phrases.

Pure numpy/scipy so it runs everywhere. Neural pitch trackers (RMVPE / FCPE) can be
plugged in through `estimate_f0(..., method=...)` once their weights are installed.
Every result carries a confidence score, and low-confidence regions are reported.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from .audio_io import resample, to_mono

ANALYSIS_SR = 16000
HOP_S = 0.01  # 10 ms frames
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Krumhansl-Kessler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def hz_to_midi(f):
    f = np.asarray(f, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f > 0, 69.0 + 12.0 * np.log2(f / 440.0), np.nan)


def midi_to_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=float) - 69.0) / 12.0)


def midi_name(m: float) -> str:
    r = int(round(m))
    return f"{NOTE_NAMES[r % 12]}{r // 12 - 1}"


# --------------------------------------------------------------------------- F0
BLOCK_FRAMES = 2000  # frames per block: keeps memory flat for long (4 min+) songs


def yin_f0(x: np.ndarray, sr: int, fmin=60.0, fmax=1100.0, hop_s=HOP_S, frame=1024, threshold=0.12):
    """Vectorised YIN, processed in blocks. Returns times, f0 (0 = unvoiced), periodicity confidence (0..1)."""
    hop = int(round(hop_s * sr))
    pad = frame // 2
    xp = np.pad(x.astype(np.float64), (pad, pad + frame))
    n_frames = 1 + (len(x)) // hop
    f0 = np.zeros(n_frames)
    conf = np.zeros(n_frames)
    for b0 in range(0, n_frames, BLOCK_FRAMES):
        b1 = min(n_frames, b0 + BLOCK_FRAMES)
        idx = np.arange(frame)[None, :] + hop * np.arange(b0, b1)[:, None]
        f0[b0:b1], conf[b0:b1] = _yin_block(xp[idx], sr, fmin, fmax, frame, threshold)
    times = np.arange(n_frames) * hop / sr
    return times, f0, conf


def _yin_block(frames, sr, fmin, fmax, frame, threshold):
    n_frames = frames.shape[0]
    w = frame // 2  # integration window
    tau_max = min(int(sr / fmin) + 2, w)
    tau_min = max(2, int(sr / fmax))
    # difference function via FFT autocorrelation of the window and energy terms
    nfft = 1 << int(np.ceil(np.log2(frame + w)))
    F = np.fft.rfft(frames, nfft, axis=1)
    Wn = np.fft.rfft(frames[:, :w], nfft, axis=1)
    corr = np.fft.irfft(F * np.conj(Wn), nfft, axis=1)[:, :tau_max]
    sq = np.cumsum(np.pad(frames ** 2, ((0, 0), (1, 0))), axis=1)
    e0 = sq[:, w] - sq[:, 0]
    taus = np.arange(tau_max)
    e_tau = sq[:, taus + w] - sq[:, taus]
    d = e0[:, None] + e_tau - 2 * corr
    d[:, 0] = 0
    d = np.maximum(d, 0)
    cum = np.cumsum(d[:, 1:], axis=1)
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] * np.arange(1, tau_max)[None, :] / np.maximum(cum, 1e-12)

    f0 = np.zeros(n_frames)
    conf = np.zeros(n_frames)
    region = cmnd[:, tau_min:]
    below = region < threshold
    has = below.any(axis=1)
    first = np.where(has, below.argmax(axis=1), region.argmin(axis=1))
    for i in range(n_frames):
        t = first[i]
        if has[i]:  # walk to the local minimum
            while t + 1 < region.shape[1] and region[i, t + 1] < region[i, t]:
                t += 1
        tau = t + tau_min
        val = cmnd[i, tau]
        if 1 <= tau < tau_max - 1:
            a, b, c = cmnd[i, tau - 1], cmnd[i, tau], cmnd[i, tau + 1]
            den = a - 2 * b + c
            shift = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
            shift = float(np.clip(shift, -1, 1))
        else:
            shift = 0.0
        conf[i] = float(np.clip(1.0 - val, 0, 1))
        if val < 0.35:
            f0[i] = sr / (tau + shift)
    return f0, conf


def frame_rms_db(x: np.ndarray, sr: int, hop_s=HOP_S, win_s=0.03):
    hop = int(round(hop_s * sr))
    win = int(round(win_s * sr))
    n_frames = 1 + len(x) // hop
    xp = np.pad(x.astype(np.float64), (win // 2, win))
    rms = np.empty(n_frames)
    for b0 in range(0, n_frames, BLOCK_FRAMES * 4):
        b1 = min(n_frames, b0 + BLOCK_FRAMES * 4)
        idx = np.arange(win)[None, :] + hop * np.arange(b0, b1)[:, None]
        rms[b0:b1] = np.sqrt(np.mean(xp[idx] ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-9)


def onset_envelope(x: np.ndarray, sr: int, hop_s=HOP_S):
    """Log-spectral flux, one value per frame."""
    hop = int(round(hop_s * sr))
    n_fft = 1024
    n_frames = 1 + len(x) // hop
    xp = np.pad(x.astype(np.float64), (n_fft // 2, n_fft))
    win = np.hanning(n_fft)
    flux = np.zeros(n_frames)
    prev = None
    for b0 in range(0, n_frames, BLOCK_FRAMES):
        b1 = min(n_frames, b0 + BLOCK_FRAMES)
        idx = np.arange(n_fft)[None, :] + hop * np.arange(b0, b1)[:, None]
        L = np.log1p(100 * np.abs(np.fft.rfft(xp[idx] * win, axis=1))).astype(np.float32)
        first = L[:1] if prev is None else prev
        flux[b0:b1] = np.maximum(np.diff(L, axis=0, prepend=first), 0).sum(axis=1)
        prev = L[-1:]
    flux = flux - uniform_filter1d(flux, 30)
    return np.maximum(flux, 0)


# ------------------------------------------------------------------------- notes
@dataclass
class Note:
    id: str
    start: float
    dur: float
    midi: float  # target pitch (fractional allowed; render rounds unless Original tuning)
    velocity: float = 0.8
    confidence: float = 1.0
    lyric: str = ""
    phonemes: dict | None = None
    continuation: bool = False  # melisma: holds previous syllable's vowel
    vibrato: dict | None = None
    sustained: bool = False
    slide_in: bool = False
    edited: bool = False

    @property
    def end(self):
        return self.start + self.dur


def segment_notes(times, f0, conf, rms_db, onset_env, min_dur=0.08, floor_db=45, min_conf=0.55):
    midi = hz_to_midi(f0)
    voiced = (f0 > 0) & (conf > min_conf) & (rms_db > np.percentile(rms_db, 99) - floor_db)
    sm = midi.copy()
    sm[~voiced] = np.nan
    # median smooth through vibrato
    # smooth over ~one vibrato cycle (150 ms) inside each voiced run so vibrato is not split into notes
    med = np.full_like(sm, np.nan)
    k = 0
    while k < len(sm):
        if np.isnan(sm[k]):
            k += 1
            continue
        j = k
        while j < len(sm) and not np.isnan(sm[j]):
            j += 1
        seg = median_filter(sm[k:j], size=3, mode="nearest")
        med[k:j] = uniform_filter1d(seg, size=min(15, j - k), mode="nearest")
        k = j

    # onset peaks (for repeated same-pitch notes)
    thr = onset_env.mean() + 2.0 * onset_env.std()
    peaks = set(
        i for i in range(1, len(onset_env) - 1)
        if onset_env[i] > thr and onset_env[i] >= onset_env[i - 1] and onset_env[i] >= onset_env[i + 1]
    )
    # rms dips also mark re-articulations of the same pitch
    rms_s = uniform_filter1d(rms_db, 5)

    notes = []
    cur = None
    hop = times[1] - times[0] if len(times) > 1 else HOP_S

    def close(c, end_i):
        s, e = c["start"], end_i
        if (e - s) * hop >= min_dur:
            seg = midi[s:e]
            seg = seg[~np.isnan(seg)]
            if len(seg):
                notes.append({"s": s, "e": e, "midi": float(np.median(seg)), "conf": float(np.mean(conf[s:e]))})

    i = 0
    n = len(times)
    while i < n:
        if not voiced[i] or np.isnan(med[i]):
            if cur is not None:
                close(cur, i)
                cur = None
            i += 1
            continue
        if cur is None:
            cur = {"start": i, "ref": med[i]}
            i += 1
            continue
        ref = np.nanmedian(midi[cur["start"]:i]) if i - cur["start"] >= 3 else cur["ref"]
        jump = abs(med[i] - ref) > 0.5
        # require the jump to persist for 5 frames (rejects vibrato / glitches)
        if jump:
            look = med[i:i + 5]
            jump = np.all(np.abs(look[~np.isnan(look)] - ref) > 0.4) and np.sum(~np.isnan(look)) >= 4
        rearticulate = (i in peaks) and (i - cur["start"]) * hop > 0.12 and (rms_s[max(0, i - 3):i].min() < rms_s[cur["start"]:i].max() - 6)
        if jump or rearticulate:
            close(cur, i)
            cur = {"start": i, "ref": med[i]}
        i += 1
    if cur is not None:
        close(cur, n)

    # merge tiny pitch splits (< 0.4 st and touching)
    merged = []
    for nt in notes:
        if merged and nt["s"] - merged[-1]["e"] <= 1 and abs(nt["midi"] - merged[-1]["midi"]) < 0.4 and (nt["s"] not in peaks):
            m = merged[-1]
            seg = midi[m["s"]:nt["e"]]
            m["e"] = nt["e"]
            m["midi"] = float(np.nanmedian(seg))
        else:
            merged.append(nt)

    out = []
    peak_db = rms_db.max()
    for k, nt in enumerate(merged):
        loud = float(np.median(rms_db[nt["s"]:nt["e"]]))
        vel = float(np.clip(1.0 + (loud - peak_db) / 40.0, 0.15, 1.0))
        out.append(Note(id=f"n{k}", start=float(times[nt["s"]]), dur=float((nt["e"] - nt["s"]) * hop),
                        midi=round(nt["midi"], 3), velocity=round(vel, 3), confidence=round(nt["conf"], 3)))
    return out


# ------------------------------------------------------------------------- tempo
def _beat_score(env, fps, bpm, phase_steps=16):
    period = 60.0 * fps / bpm
    best = 0.0
    best_phase = 0.0
    n = len(env)
    for ph in np.linspace(0, period, phase_steps, endpoint=False):
        pos = np.arange(ph, n - 1, period)
        i0 = pos.astype(int)
        w = pos - i0
        val = np.mean(env[i0] * (1 - w) + env[np.minimum(i0 + 1, n - 1)] * w)
        if val > best:
            best, best_phase = val, ph
    return best, best_phase / fps


def estimate_tempo(env: np.ndarray, note_onsets: list[float], fps=100.0, lo=60.0, hi=200.0):
    """Returns bpm, beat offset (s), confidence. Combines spectral flux and note onsets."""
    n = len(env)
    imp = np.zeros(n)
    for t in note_onsets:
        k = int(round(t * fps))
        if 0 <= k < n:
            imp[k] += 1.0
    from scipy.ndimage import gaussian_filter1d
    e = env / (env.max() + 1e-9) + gaussian_filter1d(imp, 1.5) * 2.0
    e = e - e.mean()
    if n < fps * 2 or np.allclose(e, 0):
        return 120.0, 0.0, 0.0
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    E = np.fft.rfft(e, nfft)
    ac = np.fft.irfft(E * np.conj(E), nfft)[:n]
    ac /= ac[0] + 1e-12
    cands = []
    for bpm in np.arange(lo, hi + 0.5, 0.5):
        lag = 60.0 * fps / bpm
        score = 0.0
        for mult, wt in ((1, 1.0), (2, 0.5), (4, 0.25), (0.5, 0.25)):
            L = lag * mult
            if 1 <= L < n - 1:
                i0 = int(L)
                score += wt * (ac[i0] * (1 - (L - i0)) + ac[i0 + 1] * (L - i0))
        # mild prior toward 90-130 BPM
        score *= np.exp(-0.5 * (np.log2(bpm / 110.0) / 0.9) ** 2)
        cands.append((score, bpm))
    cands.sort(reverse=True)
    coarse = cands[0][1]
    # fine search: maximise onset energy on a beat grid
    pos_e = np.maximum(e, 0)
    best = (-1.0, coarse, 0.0)
    for bpm in np.arange(coarse - 2.0, coarse + 2.0001, 0.05):
        s, ph = _beat_score(pos_e, fps, bpm)
        if s > best[0]:
            best = (s, bpm, ph)
    _, bpm, phase = best
    top = cands[0][0]
    second = next((s for s, b in cands if abs(b - coarse) > 4 and abs(b / coarse - 2) > 0.05 and abs(b / coarse - 0.5) > 0.05), 0.0)
    conf = float(np.clip((top - second) / (abs(top) + 1e-9) * 2.5, 0, 1))
    return float(round(bpm, 2)), float(phase), conf


def tempo_map(env, note_onsets, fps=100.0, window_s=8.0, global_bpm=120.0):
    """Windowed tempo to detect tempo changes."""
    n = len(env)
    win = int(window_s * fps)
    out = []
    if n < win * 1.5:
        return [{"time": 0.0, "bpm": global_bpm}]
    for s in range(0, n - win // 2, win // 2):
        seg = env[s:s + win]
        ons = [t - s / fps for t in note_onsets if s / fps <= t < (s + win) / fps]
        bpm, _, c = estimate_tempo(seg, ons, fps, lo=global_bpm * 0.8, hi=global_bpm * 1.25)
        out.append({"time": round(s / fps, 2), "bpm": bpm, "confidence": round(c, 2)})
    changes = [out[0] | {"bpm": global_bpm}]
    for seg in out[1:]:
        if abs(seg["bpm"] - changes[-1]["bpm"]) / changes[-1]["bpm"] > 0.04 and seg.get("confidence", 0) > 0.4:
            changes.append(seg)
    return changes


# --------------------------------------------------------------------------- key
def estimate_key(notes: list[Note]):
    hist = np.zeros(12)
    for n in notes:
        hist[int(round(n.midi)) % 12] += n.dur * (0.5 + n.velocity)
    if hist.sum() == 0:
        return {"tonic": "C", "mode": "major", "confidence": 0.0, "label": "C major", "short": "Cmaj"}
    scores = []
    for tonic in range(12):
        for mode, prof in (("major", _MAJOR), ("minor", _MINOR)):
            r = np.corrcoef(hist, np.roll(prof, tonic))[0, 1]
            scores.append((r, tonic, mode))
    scores.sort(reverse=True)
    r, tonic, mode = scores[0]
    conf = float(np.clip((r - scores[1][0]) * 4 + max(r, 0) * 0.5, 0, 1))
    name = NOTE_NAMES[tonic]
    return {"tonic": name, "tonic_pc": tonic, "mode": mode, "confidence": round(conf, 3),
            "label": f"{name} {mode}", "short": f"{name}{'maj' if mode == 'major' else 'min'}",
            "alternatives": [f"{NOTE_NAMES[t]} {m}" for _, t, m in scores[1:4]]}


# ----------------------------------------------------------------- expression
def detect_vibrato(times, f0, note: Note):
    hop = times[1] - times[0]
    s, e = int(note.start / hop), int(note.end / hop)
    seg = hz_to_midi(f0[s:e])
    seg = seg[~np.isnan(seg)]
    if len(seg) < 0.35 / hop:
        return None
    cents = (seg - np.polyval(np.polyfit(np.arange(len(seg)), seg, 2), np.arange(len(seg)))) * 100
    spec = np.abs(np.fft.rfft(cents * np.hanning(len(cents)), 512))
    freqs = np.fft.rfftfreq(512, hop)
    band = (freqs >= 3.5) & (freqs <= 9.0)
    if not band.any():
        return None
    k = np.argmax(spec * band)
    extent = float(np.sqrt(2) * np.std(cents))
    peak_ratio = spec[k] / (spec[1:].mean() + 1e-9)
    if extent > 15 and peak_ratio > 3:
        return {"rate_hz": round(float(freqs[k]), 2), "depth_cents": round(extent, 1)}
    return None


def detect_slides(times, f0, notes):
    hop = times[1] - times[0]
    slides = []
    for a, b in zip(notes, notes[1:]):
        gap = b.start - a.end
        if gap > 0.12 or abs(b.midi - a.midi) < 1:
            continue
        s = max(0, int((a.end - 0.1) / hop))
        e = int((b.start + 0.1) / hop)
        seg = f0[s:e]
        if len(seg) and np.mean(seg > 0) > 0.9:
            slides.append({"from": a.id, "to": b.id, "time": round(a.end, 3)})
            b.slide_in = True
    return slides


def detect_phrases(times, f0, conf, rms_db, notes, min_gap=0.25):
    hop = times[1] - times[0]
    phrases = []
    breaths = []
    if not notes:
        return phrases, breaths
    floor = np.percentile(rms_db, 10)
    cur = [notes[0]]
    for a, b in zip(notes, notes[1:]):
        gap = b.start - a.end
        if gap >= min_gap:
            phrases.append(cur)
            cur = [b]
            s, e = int(a.end / hop), int(b.start / hop)
            g = rms_db[s:e]
            unv = f0[s:e] == 0
            if len(g) and np.mean(unv) > 0.7 and np.max(g) > floor + 10:
                breaths.append({"start": round(a.end, 3), "end": round(b.start, 3)})
        else:
            cur.append(b)
    phrases.append(cur)
    return [
        {"id": f"p{i}", "start": round(p[0].start, 3), "end": round(p[-1].end, 3), "note_ids": [n.id for n in p]}
        for i, p in enumerate(phrases)
    ], breaths


def low_confidence_regions(times, conf, f0, rms_db, notes):
    hop = times[1] - times[0]
    regions = []
    active = rms_db > rms_db.max() - 35
    bad = active & ((f0 == 0) | (conf < 0.6))
    # only inside note-ish spans or loud unvoiced stretches longer than 150 ms
    i = 0
    while i < len(bad):
        if bad[i]:
            j = i
            while j < len(bad) and bad[j]:
                j += 1
            if (j - i) * hop >= 0.15:
                regions.append({"start": round(times[i], 3), "end": round(times[min(j, len(times) - 1)], 3),
                                "reason": "unstable or unvoiced pitch in a loud section"})
            i = j
        else:
            i += 1
    for n in notes:
        if n.confidence < 0.7:
            regions.append({"start": round(n.start, 3), "end": round(n.end, 3), "reason": f"low note confidence ({n.confidence:.2f})"})
    return sorted(regions, key=lambda r: r["start"])


def music_likelihood(x, sr):
    """Heuristic: is there accompaniment (needs separation)? 0..1"""
    n_fft = 2048
    hop = n_fft // 2
    if len(x) < n_fft * 4:
        return 0.0
    idx = np.arange(n_fft)[None, :] + hop * np.arange((len(x) - n_fft) // hop)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * np.hanning(n_fft), axis=1)) + 1e-9
    active = S.sum(axis=1) > np.percentile(S.sum(axis=1), 30)
    S = S[active]
    if len(S) == 0:
        return 0.0
    # count simultaneous strong peaks (polyphony) and low-frequency energy (bass/kick)
    norm = S / S.max(axis=1, keepdims=True)
    peaks = ((norm[:, 1:-1] > 0.25) & (norm[:, 1:-1] > norm[:, :-2]) & (norm[:, 1:-1] > norm[:, 2:])).sum(axis=1)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    low = S[:, freqs < 70].sum(axis=1) / S.sum(axis=1)
    score = np.clip((np.median(peaks) - 6) / 20, 0, 1) * 0.7 + np.clip(np.median(low) * 15, 0, 1) * 0.3
    return float(round(score, 3))


# ------------------------------------------------------------------- top level
@dataclass
class AnalysisResult:
    duration: float
    sample_rate: int
    hop_s: float
    f0_times: list
    f0_hz: list
    f0_conf: list
    loudness_db: list
    notes: list
    tempo: dict
    tempo_map: list
    key: dict
    phrases: list
    breaths: list
    slides: list
    low_confidence: list
    music_likelihood: float
    note_confidence: float
    warnings: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        return d


def analyze(data: np.ndarray, sr: int, progress=lambda f, m: None, separated: bool = False) -> AnalysisResult:
    """separated=True: the input came out of vocal separation, so quiet music bleed is ignored
    (stricter loudness / periodicity gates, longer minimum notes)."""
    x = to_mono(data).astype(np.float32)
    duration = len(x) / sr
    warnings = []
    if duration < 1.0:
        raise ValueError("Input is shorter than 1 second; nothing to analyse.")
    peak = float(np.max(np.abs(x)))
    if peak < 1e-4:
        raise ValueError("Input is silent.")
    if peak >= 0.999:
        warnings.append("Input appears clipped; pitch and dynamics may be less accurate.")
    progress(0.05, "Resampling")
    y = resample(x, sr, ANALYSIS_SR)
    progress(0.15, "Detecting music / accompaniment")
    ml = music_likelihood(y, ANALYSIS_SR)
    if ml > 0.5:
        warnings.append("Input seems to contain accompaniment. Run vocal separation for better results.")
    progress(0.25, "Tracking pitch (YIN)")
    times, f0, conf = yin_f0(y, ANALYSIS_SR)
    rms = frame_rms_db(y, ANALYSIS_SR)[: len(times)]
    gate = rms < rms.max() - 50
    f0[gate] = 0
    progress(0.55, "Segmenting notes")
    env = onset_envelope(y, ANALYSIS_SR)[: len(times)]
    if separated:
        notes = segment_notes(times, f0, conf, rms, env, min_dur=0.12, floor_db=24, min_conf=0.7)
        if len(notes) >= 8:  # bass / chord bleed sits far outside the singer's register
            centre = float(np.median([n.midi for n in notes]))
            kept = [n for n in notes if centre - 12.5 <= n.midi <= centre + 14]
            if len(kept) < len(notes):
                warnings.append(f"Ignored {len(notes) - len(kept)} notes far outside the singer's register (likely music left in the separated vocal).")
            notes = kept
            for i, n in enumerate(notes):
                n.id = f"n{i}"
    else:
        notes = segment_notes(times, f0, conf, rms, env)
    if not notes:
        warnings.append("No stable notes were found. Is the input pitched (hum / sing / instrument)?")
    progress(0.7, "Estimating tempo")
    bpm, beat_offset, tconf = estimate_tempo(env, [n.start for n in notes])
    tmap = tempo_map(env, [n.start for n in notes], global_bpm=bpm)
    progress(0.8, "Estimating key")
    key = estimate_key(notes)
    progress(0.85, "Expression: vibrato, slides, sustains")
    beat = 60.0 / bpm
    for n in notes:
        n.vibrato = detect_vibrato(times, f0, n)
        n.sustained = n.dur >= max(0.8, beat * 1.5)
    slides = detect_slides(times, f0, notes)
    phrases, breaths = detect_phrases(times, f0, conf, rms, notes)
    low = low_confidence_regions(times, conf, f0, rms, notes)
    note_conf = float(np.mean([n.confidence for n in notes])) if notes else 0.0
    if tconf < 0.35:
        warnings.append(f"Tempo confidence is low ({tconf:.2f}). Please confirm {bpm} BPM or tap the tempo.")
    if key["confidence"] < 0.3:
        warnings.append(f"Key confidence is low ({key['confidence']:.2f}). Alternatives: {', '.join(key.get('alternatives', []))}.")
    progress(1.0, "Done")
    r = lambda a, d=4: [round(float(v), d) for v in a]
    return AnalysisResult(
        duration=round(duration, 4), sample_rate=sr, hop_s=HOP_S,
        f0_times=r(times, 3), f0_hz=r(f0, 2), f0_conf=r(conf, 3), loudness_db=r(rms, 2),
        notes=[asdict(n) for n in notes],
        tempo={"bpm": bpm, "beat_offset": round(beat_offset, 3), "confidence": round(tconf, 3), "time_signature": [4, 4]},
        tempo_map=tmap, key=key, phrases=phrases, breaths=breaths, slides=slides,
        low_confidence=low, music_likelihood=ml, note_confidence=round(note_conf, 3), warnings=warnings,
    )
