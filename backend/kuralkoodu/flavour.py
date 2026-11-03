"""Flavour engine: deterministic, seeded, per-phrase-undoable expression applied to the
performance data layer (notes, timing, dynamics, pitch gestures) before rendering.

Rules
* Every change has a stable id "<phrase>:<kind>". Disabling one (per-phrase undo) does not
  change any other change, because each change draws from its own RNG stream
  seeded by (seed, phrase, kind).
* Melody, timing and dynamics changes are scaled by the master Creative slider.
  At Creative = 0 they are all zero, so the analysed / edited performance is kept.
* Breathiness and grit are timbre settings (they do not move notes) and apply at any Creative value.
"""
from __future__ import annotations

import copy
import zlib

import numpy as np

CONTROLS = ["vibrato", "scoops", "slides", "melisma", "breathiness", "grit", "timing", "dynamics", "gamaka"]

PRESETS = {
    "Natural": dict(vibrato=35, scoops=15, slides=20, melisma=0, breathiness=15, grit=0, timing=10, dynamics=30, gamaka=0),
    "Pop": dict(vibrato=40, scoops=30, slides=25, melisma=15, breathiness=25, grit=5, timing=10, dynamics=45, gamaka=0),
    "R&B": dict(vibrato=55, scoops=45, slides=45, melisma=60, breathiness=30, grit=5, timing=35, dynamics=55, gamaka=0),
    "Soul": dict(vibrato=60, scoops=40, slides=40, melisma=45, breathiness=25, grit=35, timing=30, dynamics=70, gamaka=0),
    "Rock": dict(vibrato=30, scoops=35, slides=20, melisma=5, breathiness=10, grit=65, timing=15, dynamics=65, gamaka=0),
    "Cinematic": dict(vibrato=50, scoops=10, slides=35, melisma=10, breathiness=35, grit=0, timing=5, dynamics=80, gamaka=0),
    "Indian Classical-inspired": dict(vibrato=20, scoops=20, slides=60, melisma=50, breathiness=10, grit=0, timing=10, dynamics=50, gamaka=80),
}

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    # Carnatic ragas (arohanam pitch sets, simplified to 12-TET)
    "Shankarabharanam": [0, 2, 4, 5, 7, 9, 11],
    "Kalyani": [0, 2, 4, 6, 7, 9, 11],
    "Kharaharapriya": [0, 2, 3, 5, 7, 9, 10],
    "Natabhairavi": [0, 2, 3, 5, 7, 8, 10],
    "Mayamalavagowla": [0, 1, 4, 5, 7, 8, 11],
    "Harikambhoji": [0, 2, 4, 5, 7, 9, 10],
    "Mohanam": [0, 2, 4, 7, 9],
    "Hindolam": [0, 3, 5, 8, 10],
    "Hamsadhwani": [0, 2, 4, 7, 11],
    "Sivaranjani": [0, 2, 3, 7, 9],
}

DEFAULT_SETTINGS = {
    "creative": 40, "preset": "Natural", **PRESETS["Natural"],
    "tuning": "natural", "scale": None, "seed": 1234, "seed_locked": True,
    "doubles": {"enabled": False, "count": 1}, "harmonies": {"enabled": False, "intervals": [2]},
    "adlibs": {"enabled": False}, "disabled_changes": [],
}


def scale_pcs(key: dict, scale_name: str | None):
    name = scale_name or key.get("mode", "major")
    steps = SCALES.get(name, SCALES["major"])
    tonic = key.get("tonic_pc", 0)
    return sorted({(tonic + s) % 12 for s in steps})


def snap_to_scale(m: float, pcs):
    r = int(round(m))
    best = min(range(r - 2, r + 3), key=lambda c: (abs(c - m) + (0 if c % 12 in pcs else 99)))
    return best


def scale_step(m: int, steps: int, pcs):
    """Move `steps` scale degrees from midi note m (m snapped first)."""
    cur = snap_to_scale(m, pcs)
    d = 1 if steps > 0 else -1
    for _ in range(abs(steps)):
        cur += d
        while cur % 12 not in pcs:
            cur += d
    return cur


def _rng(seed, phrase_id, kind):
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, zlib.crc32(phrase_id.encode()), zlib.crc32(kind.encode())])


def apply_flavours(perf: dict, s: dict):
    """Return (new_perf, changes). perf['notes'] is not modified in place."""
    s = {**DEFAULT_SETTINGS, **(s or {})}
    p = copy.deepcopy(perf)
    notes = sorted(p["notes"], key=lambda n: n["start"])
    p["notes"] = notes
    by_id = {n["id"]: n for n in notes}
    phrases = p.get("phrases") or [{"id": "p0", "note_ids": [n["id"] for n in notes]}]
    creative = float(s["creative"]) / 100.0
    eff = {c: float(s.get(c, 0)) / 100.0 * creative for c in CONTROLS}
    disabled = set(s.get("disabled_changes") or [])
    seed = int(s["seed"])
    pcs = scale_pcs(p.get("key", {}), s.get("scale"))
    changes = []
    beat = 60.0 / float(p.get("tempo", {}).get("bpm", 120))

    for n in notes:
        n.setdefault("fx", {})
        n["fx"] = {"vibrato": None, "scoop": None, "slide_ms": 0.0, "timing_ms": 0.0, "vel_mult": 1.0, "gamaka": None}

    def record(ph, kind, summary, ids, detail=None):
        cid = f"{ph['id']}:{kind}"
        changes.append({"id": cid, "phrase": ph["id"], "kind": kind, "summary": summary, "note_ids": ids,
                        "enabled": cid not in disabled, "detail": detail or {}})
        return cid not in disabled

    new_notes = []
    for ph in phrases:
        pn = [by_id[i] for i in ph["note_ids"] if i in by_id]
        if not pn:
            continue
        # --- vibrato on sustained notes
        if eff["vibrato"] > 0:
            r = _rng(seed, ph["id"], "vibrato")
            tgt = [n for n in pn if n["dur"] >= 0.35 and not n.get("vibrato")]
            plan = []
            for n in tgt:
                depth = 15 + 65 * eff["vibrato"] * float(r.uniform(0.8, 1.2))
                rate = float(r.uniform(4.8, 6.2))
                plan.append((n, {"depth": round(depth, 1), "rate": round(rate, 2), "delay": round(min(0.3, 0.3 * n["dur"]), 3)}))
            if plan and record(ph, "vibrato", f"Vibrato on {len(plan)} sustained note(s), ~{np.mean([v['depth'] for _, v in plan]):.0f} cents", [n["id"] for n, _ in plan]):
                for n, v in plan:
                    n["fx"]["vibrato"] = v
        # --- gamaka (kampita: oscillation toward the upper scale neighbour)
        if eff["gamaka"] > 0:
            r = _rng(seed, ph["id"], "gamaka")
            plan = []
            for n in pn:
                if n["dur"] >= 0.4 and r.random() < 0.35 + 0.6 * eff["gamaka"]:
                    up = scale_step(int(round(n["midi"])), 1, pcs) - round(n["midi"])
                    plan.append((n, {"interval": up * 100 * (0.4 + 0.5 * eff["gamaka"]), "rate": float(r.uniform(2.8, 4.2)), "delay": 0.12}))
            if plan and record(ph, "gamaka", f"Gamaka (kampita) on {len(plan)} note(s)", [n["id"] for n, _ in plan]):
                for n, g in plan:
                    n["fx"]["gamaka"] = g
                    n["fx"]["vibrato"] = None
        # --- scoops into phrase starts and after leaps
        if eff["scoops"] > 0:
            r = _rng(seed, ph["id"], "scoops")
            plan = []
            for i, n in enumerate(pn):
                leap = i > 0 and abs(n["midi"] - pn[i - 1]["midi"]) >= 3
                if (i == 0 or leap) and r.random() < 0.3 + 0.7 * eff["scoops"]:
                    plan.append((n, {"cents": round(-(40 + 160 * eff["scoops"]) * float(r.uniform(0.7, 1.1)), 1), "ms": round(float(r.uniform(60, 130)), 1)}))
            if plan and record(ph, "scoops", f"Pitch scoops into {len(plan)} note(s)", [n["id"] for n, _ in plan]):
                for n, sc in plan:
                    n["fx"]["scoop"] = sc
        # --- slides (portamento) between connected notes
        if eff["slides"] > 0:
            r = _rng(seed, ph["id"], "slides")
            plan = []
            for a, b in zip(pn, pn[1:]):
                if b["start"] - (a["start"] + a["dur"]) < 0.09 and a["midi"] != b["midi"] and r.random() < 0.2 + 0.8 * eff["slides"]:
                    plan.append((b, round(40 + 180 * eff["slides"] * float(r.uniform(0.6, 1.2)), 1)))
            if plan and record(ph, "slides", f"Slides into {len(plan)} note(s)", [n["id"] for n, _ in plan]):
                for n, ms in plan:
                    n["fx"]["slide_ms"] = ms
        # --- timing looseness (laid back, consistent per phrase)
        if eff["timing"] > 0:
            r = _rng(seed, ph["id"], "timing")
            bias = float(r.normal(8, 6)) * eff["timing"]
            offs = [float(np.clip(bias + r.normal(0, 22 * eff["timing"]), -45, 60)) for _ in pn]
            if record(ph, "timing", f"Timing loosened (avg {np.mean(np.abs(offs)):.0f} ms)", [n["id"] for n in pn]):
                for n, o in zip(pn, offs):
                    n["fx"]["timing_ms"] = round(o, 1)
        # --- dynamic expression: phrase arc + downbeat accents
        if eff["dynamics"] > 0:
            t0, t1 = pn[0]["start"], pn[-1]["start"] + pn[-1]["dur"]
            L = max(t1 - t0, 1e-3)
            if record(ph, "dynamics", "Phrase arc (swell then release)", [n["id"] for n in pn]):
                for n in pn:
                    x = (n["start"] - t0) / L
                    arc = np.sin(np.pi * min(1.0, x / 0.7) * 0.5) if x < 0.7 else 1 - (x - 0.7) / 0.3 * 0.5
                    n["fx"]["vel_mult"] = round(float(1 + 0.4 * eff["dynamics"] * (arc - 0.6)), 3)
        # --- melisma: ornament a phrase-final sustained note with a scale turn
        if eff["melisma"] > 0:
            r = _rng(seed, ph["id"], "melisma")
            last = pn[-1]
            if last["dur"] >= 0.45 and r.random() < 0.25 + 0.75 * eff["melisma"]:
                k = 2 + int(round(3 * eff["melisma"] * float(r.uniform(0.6, 1.0))))
                orn_len = last["dur"] * 0.55
                each = orn_len / (k + 1)
                if each >= 0.07:
                    base = int(round(last["midi"]))
                    path = [1, 0, -1, 0, 1, 2, 1, 0][:k]
                    if record(ph, "melisma", f"Melisma turn ({k} notes) at phrase end", [last["id"]]):
                        start = last["start"] + last["dur"] - orn_len
                        last["dur"] = round(start - last["start"], 4)
                        prev_coda = (last.get("phonemes") or {}).get("coda", [])
                        if last.get("phonemes"):
                            last["phonemes"] = {**last["phonemes"], "coda": []}
                        vowel = (last.get("phonemes") or {}).get("vowel", "a")
                        for j, stp in enumerate(path + [0]):
                            mm = scale_step(base, stp, pcs) if stp else base
                            new_notes.append({
                                "id": f"{last['id']}m{j}", "start": round(start + j * each, 4), "dur": round(each, 4),
                                "midi": float(mm), "velocity": last.get("velocity", 0.8) * 0.9, "confidence": 1.0,
                                "lyric": "-", "continuation": True, "edited": False, "generated": "melisma",
                                "phonemes": {"onset": [], "vowel": vowel, "coda": prev_coda if j == len(path) else []},
                                "fx": {"vibrato": None, "scoop": None, "slide_ms": 30.0, "timing_ms": last["fx"]["timing_ms"], "vel_mult": last["fx"]["vel_mult"], "gamaka": None},
                            })
    p["notes"] = sorted(notes + new_notes, key=lambda n: n["start"])
    p["timbre"] = {"breathiness": float(s.get("breathiness", 0)) / 100.0, "grit": float(s.get("grit", 0)) / 100.0}
    p["tuning"] = s.get("tuning", "natural")
    p["scale_pcs"] = pcs
    return p, changes


# ----------------------------------------------------------- extra parts
def make_double(perf: dict, seed: int, index: int):
    r = np.random.default_rng([seed, 9000 + index])
    p = copy.deepcopy(perf)
    for n in p["notes"]:
        fx = n.setdefault("fx", {})
        fx["timing_ms"] = float(fx.get("timing_ms", 0.0) + r.normal(0, 12))
        n["detune_cents"] = float(r.normal(0, 7))
    p["timbre_offset"] = {"formant": float(r.uniform(-0.03, 0.03)), "seed_offset": 101 + index}
    return p


def make_harmony(perf: dict, steps: int):
    pcs = perf.get("scale_pcs") or scale_pcs(perf.get("key", {}), None)
    p = copy.deepcopy(perf)
    for n in p["notes"]:
        n["midi"] = float(scale_step(int(round(n["midi"])), steps, pcs))
    p["timbre_offset"] = {"formant": 0.01 * np.sign(steps), "seed_offset": 200 + steps}
    return p


ADLIB_VOCABLES = {"ta": [("ஆ", "aː", []), ("ம்", "m", []), ("ஓ", "oː", []), ("லா", "aː", ["l"])],
                  "en": [("ooh", "uː", []), ("yeah", "e", ["j"]), ("oh", "oː", []), ("mm", "m", [])]}


def make_adlibs(perf: dict, seed: int, language="ta"):
    """Short fills in phrase gaps >= 0.6 s, on its own stem."""
    r = np.random.default_rng([seed, 777])
    pcs = perf.get("scale_pcs") or scale_pcs(perf.get("key", {}), None)
    notes = sorted(perf["notes"], key=lambda n: n["start"])
    voc = ADLIB_VOCABLES.get("en" if language == "en" else "ta")
    out = []
    for a, b in zip(notes, notes[1:]):
        gap = b["start"] - (a["start"] + a["dur"])
        if gap < 0.6:
            continue
        text, vowel, onset = voc[int(r.integers(len(voc)))]
        k = int(r.integers(2, 4))
        t = a["start"] + a["dur"] + 0.08
        each = min(0.22, (gap - 0.16) / k)
        base = int(round(a["midi"]))
        for j in range(k):
            stp = int(r.choice([-1, 0, 1, 2])) if j else 1
            nm = {"id": f"ad{len(out)}", "start": round(t + j * each, 4), "dur": round(each * 0.95, 4),
                  "midi": float(scale_step(base, stp, pcs)), "velocity": 0.55, "lyric": text if j == 0 else "-",
                  "continuation": j > 0, "phonemes": {"onset": onset if j == 0 else [], "vowel": vowel if vowel != "m" else "u", "coda": ["m"] if vowel == "m" else []},
                  "fx": {"vibrato": None, "scoop": {"cents": -80, "ms": 60} if j == 0 else None, "slide_ms": 40, "timing_ms": 0, "vel_mult": 1, "gamaka": None}}
            out.append(nm)
    p = copy.deepcopy(perf)
    p["notes"] = out
    p["timbre_offset"] = {"formant": 0.0, "seed_offset": 300}
    return p
