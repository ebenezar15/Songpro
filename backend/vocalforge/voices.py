"""Voice bank (built-in, parametric, owned by this project) and age/gender timbre presets.

Age/gender are implemented as calibrated changes to vocal-tract length (formant scale),
spectral tilt (brightness), aspiration (breathiness), jitter/shimmer and vibrato habits,
not by pitch shifting. Neural engines receive the same preset as a reference-voice choice.
"""
from __future__ import annotations

from .analysis import midi_name

AGES = ["child", "teen", "young_adult", "adult", "mature", "senior"]
AGE_LABELS = {"child": "Child-like tone", "teen": "Teen", "young_adult": "Young Adult",
              "adult": "Adult", "mature": "Mature", "senior": "Senior"}
GENDERS = ["male", "female"]

# formant_scale, brightness(0-1), breath(0-1), jitter(cents), vib_rate, vib_depth(cents), range(lo, hi) MIDI
_AGE = {
    ("male", "child"): (1.30, 0.75, 0.25, 4, 5.8, 30, (60, 76)),
    ("male", "teen"): (1.08, 0.65, 0.22, 5, 5.6, 35, (50, 69)),
    ("male", "young_adult"): (1.00, 0.60, 0.15, 4, 5.5, 40, (47, 67)),
    ("male", "adult"): (0.98, 0.55, 0.15, 5, 5.3, 45, (45, 65)),
    ("male", "mature"): (0.96, 0.48, 0.20, 7, 5.0, 50, (43, 64)),
    ("male", "senior"): (0.95, 0.40, 0.32, 12, 4.6, 65, (43, 62)),
    ("female", "child"): (1.32, 0.78, 0.25, 4, 5.9, 30, (60, 77)),
    ("female", "teen"): (1.20, 0.70, 0.24, 5, 5.7, 35, (57, 77)),
    ("female", "young_adult"): (1.17, 0.66, 0.20, 4, 5.6, 40, (55, 77)),
    ("female", "adult"): (1.15, 0.60, 0.20, 5, 5.4, 45, (53, 74)),
    ("female", "mature"): (1.12, 0.52, 0.24, 7, 5.1, 50, (52, 72)),
    ("female", "senior"): (1.10, 0.45, 0.33, 12, 4.7, 65, (50, 70)),
}

VOICE_BANK = [
    {"id": "kavi", "name": "Kavi", "gender": "male", "desc": "warm, rounded tenor", "formant": 1.0, "bright": 0.0, "breath": 0.0},
    {"id": "arun", "name": "Arun", "gender": "male", "desc": "bright, forward baritone", "formant": 0.97, "bright": 0.12, "breath": -0.03},
    {"id": "meera", "name": "Meera", "gender": "female", "desc": "clear, bright soprano", "formant": 1.02, "bright": 0.1, "breath": -0.03},
    {"id": "nila", "name": "Nila", "gender": "female", "desc": "soft, breathy alto", "formant": 0.98, "bright": -0.08, "breath": 0.12},
]
LICENSE_NOTE = "Built-in parametric voices are synthesised from code in this repository (no recorded voice data)."


def voice_params(gender="female", age="young_adult", voice_id=None, reference=None):
    key = (gender, age)
    if key not in _AGE:
        raise ValueError(f"Unknown voice preset {key}")
    fs, br, bt, jit, vr, vd, rng = _AGE[key]
    v = next((b for b in VOICE_BANK if b["id"] == voice_id), None)
    if v:
        fs *= v["formant"]
        br = min(1, max(0, br + v["bright"]))
        bt = min(1, max(0, bt + v["breath"]))
    p = {"gender": gender, "age": age, "voice_id": voice_id, "formant_scale": round(fs, 3), "brightness": br,
         "breathiness": bt, "jitter_cents": jit, "vibrato_rate": vr, "vibrato_depth": vd,
         "range": {"low": rng[0], "high": rng[1], "low_name": midi_name(rng[0]), "high_name": midi_name(rng[1])},
         "source": "voice_bank"}
    if reference:  # user-supplied reference: blend its measured timbre in
        p["formant_scale"] = round(0.5 * p["formant_scale"] + 0.5 * reference.get("formant_scale", fs), 3)
        p["breathiness"] = round(0.5 * p["breathiness"] + 0.5 * reference.get("breathiness", bt), 3)
        p["brightness"] = round(0.5 * p["brightness"] + 0.5 * reference.get("brightness", br), 3)
        p["source"] = "user_reference"
    return p


def range_check(notes, vp):
    if not notes:
        return {"ok": True, "message": "No notes."}
    import numpy as np
    pitches = np.array([n["midi"] for n in notes])
    lo, hi = float(np.percentile(pitches, 5)), float(np.percentile(pitches, 95))
    rlo, rhi = vp["range"]["low"], vp["range"]["high"]
    out_low = [n["id"] for n in notes if n["midi"] < rlo - 0.5]
    out_high = [n["id"] for n in notes if n["midi"] > rhi + 0.5]
    centre_m = (lo + hi) / 2
    centre_r = (rlo + rhi) / 2
    semis = int(round(centre_r - centre_m))
    octave = int(round(semis / 12)) * 12
    def fits(shift):
        return lo + shift >= rlo - 0.5 and hi + shift <= rhi + 0.5
    suggestions = []
    if octave and fits(octave):
        suggestions.append({"semitones": octave, "label": f"Octave {'up' if octave > 0 else 'down'}"})
    if semis and semis != octave:
        suggestions.append({"semitones": semis, "label": f"Transpose {semis:+d} semitones" + (" (fits)" if fits(semis) else " (closest)")})
    ok = not out_low and not out_high
    msg = ("Melody sits within the comfortable range." if ok else
           f"{len(out_low) + len(out_high)} notes are outside the comfortable range "
           f"{vp['range']['low_name']}–{vp['range']['high_name']} (melody {midi_name(lo)}–{midi_name(hi)}).")
    return {"ok": ok, "message": msg, "melody_low": midi_name(lo), "melody_high": midi_name(hi),
            "out_of_range_low": out_low, "out_of_range_high": out_high, "suggestions": suggestions}


def analyse_reference(x, sr):
    """Rough timbre measurement of a user reference voice: formant scale, breathiness, brightness."""
    import numpy as np
    from .audio_io import resample, to_mono
    y = resample(to_mono(x), sr, 16000)
    n = 1024
    hop = 256
    if len(y) < n * 8:
        raise ValueError("Reference voice is too short (need at least ~1 second of singing or speech).")
    idx = np.arange(n)[None, :] + hop * np.arange((len(y) - n) // hop)[:, None]
    fr = y[idx] * np.hanning(n)
    S = np.abs(np.fft.rfft(fr, axis=1))
    e = S.sum(axis=1)
    S = S[e > np.percentile(e, 60)]
    freqs = np.fft.rfftfreq(n, 1 / 16000)
    cent = float(np.median((S * freqs).sum(axis=1) / (S.sum(axis=1) + 1e-9)))
    # spectral centroid of male reference ~1400 Hz, female ~1800 Hz (rough)
    fscale = float(np.clip(cent / 1450.0, 0.9, 1.35))
    hf = float(np.median(S[:, freqs > 4000].sum(axis=1) / (S.sum(axis=1) + 1e-9)))
    return {"formant_scale": round(fscale, 3), "breathiness": round(float(np.clip(hf * 8, 0, 0.6)), 3),
            "brightness": round(float(np.clip((cent - 1000) / 1500, 0, 1)), 3), "centroid_hz": round(cent, 1)}
