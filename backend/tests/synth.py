"""Generated test audio with known tempo, pitch and loudness."""
import numpy as np


def hum(midis, bpm=120.0, sr=44100, beats_per_note=1.0, lead_in=0.5, gap=0.04, loud_db=None, seed=0, vibrato=0.0):
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    total = lead_in + len(midis) * beats_per_note * beat + 0.5
    y = np.zeros(int(total * sr))
    onsets = []
    for i, m in enumerate(midis):
        st = lead_in + i * beats_per_note * beat
        dur = beats_per_note * beat - gap
        n = int(dur * sr)
        t = np.arange(n) / sr
        f = 440 * 2 ** ((m - 69) / 12) * 2 ** (vibrato / 1200 * np.sin(2 * np.pi * 5.5 * t))
        ph = 2 * np.pi * np.cumsum(f) / sr
        tone = sum(np.sin(k * ph) / k ** 1.5 for k in range(1, 8))
        env = np.minimum(1, t / 0.02) * np.minimum(1, (dur - t) / 0.03)
        a = 10 ** ((loud_db[i] if loud_db is not None else -12) / 20)
        s = int(st * sr)
        y[s:s + n] += a * env * tone / 2
        onsets.append(st)
    y += rng.normal(0, 1e-4, len(y))
    return y.astype(np.float32), sr, onsets
