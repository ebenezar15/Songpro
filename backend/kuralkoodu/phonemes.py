"""Acoustic targets for the local formant singer (adult-male reference; scaled per voice).

Formants F1..F5 in Hz, bandwidths in Hz. Retroflexes (ʈ ɖ ɳ ɭ ɻ) get the lowered F3/F4
that listeners use as the main retroflex cue; Tamil stops are unaspirated.
"""
from __future__ import annotations

B_DEFAULT = (80, 100, 140, 220, 280)


def V(f1, f2, f3=2550, f4=3400, f5=4000, dur=None):
    return {"kind": "vowel", "F": (f1, f2, f3, f4, f5), "B": B_DEFAULT, "voiced": 1.0, "noise": 0.0}


VOWELS = {
    "a": V(680, 1300), "aː": V(740, 1220), "i": V(310, 2150, 2850), "iː": V(280, 2300, 2950),
    "u": V(350, 900, 2350), "uː": V(320, 800, 2300), "e": V(450, 1900, 2600), "eː": V(420, 2050, 2700),
    "o": V(480, 950, 2450), "oː": V(450, 860, 2400), "ɨ": V(370, 1450, 2450), "ə": V(520, 1400),
    "ɛ": V(560, 1800), "ɔ": V(580, 900, 2450), "ɔi": V(580, 900, 2450),
}
# diphthongs: (start vowel, end vowel)
DIPHTHONGS = {"ai": ("a", "i"), "au": ("a", "u"), "ɔi": ("ɔ", "i")}
VOWELS["ai"] = VOWELS["a"]
VOWELS["au"] = VOWELS["a"]


def C(kind, F, voiced, noise=0.0, ncenter=3000, nbw=2000, dur=0.07, gain=1.0, B=None, nasal=False):
    return {"kind": kind, "F": F, "B": B or B_DEFAULT, "voiced": voiced, "noise": noise,
            "ncenter": ncenter, "nbw": nbw, "dur": dur, "gain": gain, "nasal": nasal}


_N = (250, 1100, 2300, 3300, 4000)
CONSONANTS = {
    # stops: voiced=0 => silent closure then burst; Tamil: no aspiration
    "k": C("stop", (300, 1800, 2400, 3400, 4000), 0.0, 1.0, 2000, 1200, 0.07),
    "g": C("stop", (300, 1800, 2400, 3400, 4000), 0.35, 0.6, 2000, 1200, 0.06),
    "ʈ": C("stop", (350, 1700, 1900, 2900, 3800), 0.0, 1.0, 2700, 1400, 0.065),
    "ɖ": C("stop", (350, 1700, 1900, 2900, 3800), 0.4, 0.5, 2700, 1400, 0.045),
    "t̪": C("stop", (350, 1500, 2600, 3500, 4100), 0.0, 1.0, 4500, 2500, 0.065),
    "d̪": C("stop", (350, 1500, 2600, 3500, 4100), 0.4, 0.5, 4500, 2500, 0.05),
    "t": C("stop", (350, 1700, 2650, 3500, 4100), 0.0, 1.0, 4000, 2500, 0.065),
    "d": C("stop", (350, 1700, 2650, 3500, 4100), 0.4, 0.5, 4000, 2500, 0.05),
    "p": C("stop", (300, 900, 2300, 3300, 4000), 0.0, 0.8, 1200, 1600, 0.07),
    "b": C("stop", (300, 900, 2300, 3300, 4000), 0.4, 0.4, 1200, 1600, 0.055),
    "tʃ": C("affricate", (300, 2000, 2700, 3500, 4100), 0.0, 1.0, 3600, 1800, 0.09),
    "dʒ": C("affricate", (300, 2000, 2700, 3500, 4100), 0.45, 0.7, 3600, 1800, 0.075),
    # fricatives
    "s": C("fric", (350, 1700, 2700, 3500, 4100), 0.0, 1.0, 6500, 3000, 0.09),
    "z": C("fric", (350, 1700, 2700, 3500, 4100), 0.5, 0.6, 6000, 3000, 0.08),
    "ʃ": C("fric", (350, 1900, 2500, 3400, 4000), 0.0, 1.0, 3500, 2000, 0.095),
    "ʂ": C("fric", (350, 1700, 2000, 3000, 3900), 0.0, 1.0, 3000, 1600, 0.095),
    "f": C("fric", (350, 1100, 2400, 3400, 4000), 0.0, 0.5, 5500, 5000, 0.08),
    "v": C("fric", (350, 1100, 2400, 3400, 4000), 0.6, 0.3, 4500, 4000, 0.06),
    "θ": C("fric", (350, 1400, 2600, 3500, 4100), 0.0, 0.4, 6000, 5000, 0.08),
    "ð": C("approx", (380, 1450, 2550, 3500, 4100), 0.85, 0.15, 5000, 4000, 0.045, gain=0.6),
    "h": C("fric", None, 0.0, 0.5, 1500, 3000, 0.07),  # aspiration takes neighbour formants
    "ɣ": C("approx", (420, 1500, 2400, 3400, 4000), 0.8, 0.25, 1800, 1500, 0.045, gain=0.6),
    "x": C("fric", (400, 1500, 2400, 3400, 4000), 0.0, 0.7, 1800, 1500, 0.08),
    "β": C("approx", (380, 900, 2300, 3300, 4000), 0.85, 0.1, 1500, 2000, 0.045, gain=0.6),
    # nasals
    "m": C("nasal", (250, 1000, 2200, 3300, 4000), 1.0, 0, dur=0.075, gain=0.5, nasal=True),
    "n": C("nasal", (250, 1500, 2550, 3400, 4000), 1.0, 0, dur=0.07, gain=0.5, nasal=True),
    "n̪": C("nasal", (250, 1400, 2600, 3400, 4000), 1.0, 0, dur=0.07, gain=0.5, nasal=True),
    "ɳ": C("nasal", (250, 1500, 1950, 2900, 3800), 1.0, 0, dur=0.07, gain=0.5, nasal=True),
    "ɲ": C("nasal", (250, 2000, 2700, 3400, 4000), 1.0, 0, dur=0.07, gain=0.5, nasal=True),
    "ŋ": C("nasal", (250, 1900, 2300, 3300, 4000), 1.0, 0, dur=0.07, gain=0.5, nasal=True),
    # liquids / approximants
    "l": C("approx", (360, 1100, 2700, 3500, 4100), 1.0, 0, dur=0.055, gain=0.75),
    "ɭ": C("approx", (380, 1300, 2000, 2900, 3800), 1.0, 0, dur=0.06, gain=0.75),
    "ɻ": C("approx", (420, 1300, 1650, 2700, 3700), 1.0, 0, dur=0.065, gain=0.8),
    "ɾ": C("tap", (420, 1500, 2000, 3200, 4000), 1.0, 0, dur=0.03, gain=0.55),
    "r": C("trill", (420, 1400, 1900, 3200, 4000), 1.0, 0.05, dur=0.075, gain=0.7),
    "j": C("approx", (300, 2150, 2900, 3500, 4100), 1.0, 0, dur=0.05, gain=0.8),
    "w": C("approx", (330, 700, 2300, 3300, 4000), 1.0, 0, dur=0.05, gain=0.75),
    "ʋ": C("approx", (350, 1050, 2350, 3300, 4000), 1.0, 0.03, dur=0.05, gain=0.75),
}


def lookup(ph: str):
    if ph in VOWELS:
        return VOWELS[ph]
    if ph in CONSONANTS:
        return CONSONANTS[ph]
    # multi-char fallbacks like 'ks'
    return CONSONANTS.get(ph[:1], CONSONANTS["h"])


def split_phones(seq):
    """['ks'] -> ['k','s'] for compound romanised units."""
    out = []
    for p in seq:
        if p in CONSONANTS or p in VOWELS:
            out.append(p)
        elif p == "ks":
            out += ["k", "s"]
        elif p == "kʂ":
            out += ["k", "ʂ"]
        elif p:
            out.append(p[0] if p[0] in CONSONANTS else "h")
    return out
