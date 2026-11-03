"""Synthetic songs for separation tests: a centre-panned vocal over a repeating stereo band."""
import numpy as np

from vocalforge.engines import FormantSingerEngine
from vocalforge.lyrics import align_to_notes, parse_lyrics
from vocalforge.planner import build_plan
from vocalforge.voices import voice_params


def band(seconds, sr=44100, bpm=100, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    beat = 60 / bpm
    out = np.zeros((n, 2), np.float32)
    chords = [[48, 52, 55], [45, 48, 52], [41, 45, 48], [43, 47, 50]]  # C Am F G, one bar each
    bar = 4 * beat
    for i in range(int(seconds / bar) + 1):
        s, e = int(i * bar * sr), min(n, int((i + 1) * bar * sr))
        if s >= n:
            break
        tt = t[s:e] - t[s]
        env = np.exp(-tt * 0.8)
        for j, m in enumerate(chords[i % 4]):
            f = 440 * 2 ** ((m + 12 - 69) / 12)
            tone = sum(np.sin(2 * np.pi * f * k * tt) / k ** 1.3 for k in range(1, 7)) * env * 0.08
            pan = [-0.8, 0.0, 0.8][j]
            out[s:e, 0] += tone * (1 - pan) / 2 * 1.4
            out[s:e, 1] += tone * (1 + pan) / 2 * 1.4
        fb = 440 * 2 ** ((chords[i % 4][0] - 12 - 69) / 12)  # bass, centre
        bass = np.sin(2 * np.pi * fb * tt) * 0.18 * np.exp(-tt * 0.3)
        out[s:e] += bass[:, None]
    kick = np.exp(-np.arange(int(0.15 * sr)) / (0.03 * sr)) * np.sin(2 * np.pi * 55 * np.arange(int(0.15 * sr)) / sr) * 0.5
    hat = rng.normal(0, 1, int(0.04 * sr)) * np.exp(-np.arange(int(0.04 * sr)) / (0.008 * sr)) * 0.06
    for b in np.arange(0, seconds, beat):
        k0 = int(b * sr)
        out[k0:k0 + len(kick)] += kick[: max(0, min(len(kick), n - k0)), None]
        h0 = int((b + beat / 2) * sr)
        L = max(0, min(len(hat), n - h0))
        out[h0:h0 + L, 0] += hat[:L] * 0.7
        out[h0:h0 + L, 1] += hat[:L] * 1.0
    return out


def vocal(seconds, sr_out=44100, bpm=100, seed=0):
    rng = np.random.default_rng(seed)
    beat = 60 / bpm
    notes, t = [], 1.0
    while t < seconds - 1.5:
        d = beat * float(rng.choice([1, 1, 2]))
        notes.append({"id": f"n{len(notes)}", "start": t, "dur": d * 0.92, "midi": float(rng.choice([64, 65, 67, 69, 71, 72])), "velocity": 0.8})
        t += d
        if rng.random() < 0.15:
            t += beat * 2  # phrase gap
    syl = parse_lyrics(" ".join(["கண்ணே கலைமானே கன்னி மயிலென கண்டேன் உன்னை நானே"] * 40), "ta")
    notes, _ = align_to_notes(notes, syl)
    plan = build_plan({"notes": notes, "tuning": "natural"}, voice_params("female", "young_adult"), seed=seed, tail=0.0)
    y, sr = FormantSingerEngine().render(plan, seed)
    from vocalforge.audio_io import resample
    y = resample(y, sr, sr_out)
    n = int(seconds * sr_out)
    y = np.pad(y, (0, max(0, n - len(y))))[:n]
    return (y / (np.abs(y).max() + 1e-9) * 0.35).astype(np.float32)


def song(seconds=30, sr=44100, seed=0):
    v = vocal(seconds, sr, seed=seed)
    b = band(seconds, sr, seed=seed)
    return b + v[:, None], v, b
