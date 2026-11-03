"""Turns a (flavoured) performance into a frame-level render plan shared by all engines:
pitch track, amplitude track, and a phoneme timeline. User edits always win: edited notes
use their edited pitch/timing and ignore the detected contour."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .analysis import hz_to_midi, midi_to_hz
from .flavour import snap_to_scale
from .phonemes import DIPHTHONGS, lookup, split_phones

FPS = 200  # plan frame rate (5 ms)
TUNING_DEVIATION = {"original": 1.0, "natural": 0.35, "hard": 0.0}


def _note_times(notes):
    out = []
    prev_end = -1.0
    for n in notes:
        st = n["start"] + float(n.get("fx", {}).get("timing_ms", 0.0)) / 1000.0
        st = max(st, prev_end - 0.005, 0.0)
        en = st + max(0.03, n["dur"])
        out.append((st, en))
        prev_end = en
    return out


def build_plan(perf: dict, voice: dict, analysis: dict | None = None, window: tuple[float, float] | None = None,
               seed: int = 0, tail: float = 0.6):
    notes = sorted(perf["notes"], key=lambda n: n["start"])
    tuning = perf.get("tuning", "natural")
    transpose = float(perf.get("transpose", 0))
    pcs = perf.get("scale_pcs")
    dur = (max((n["start"] + n["dur"] for n in notes), default=1.0) if notes else 1.0) + tail
    if analysis:
        dur = max(dur, analysis.get("duration", 0) + 0.2)
    n_frames = int(np.ceil(dur * FPS)) + 1
    t = np.arange(n_frames) / FPS
    midi = np.full(n_frames, np.nan)
    amp = np.zeros(n_frames)
    owner = np.full(n_frames, -1)
    times = _note_times(notes)

    # input contour for "Original"/"Natural" tuning
    in_t = in_m = in_db = None
    if analysis and analysis.get("f0_times"):
        in_t = np.asarray(analysis["f0_times"])
        in_m = hz_to_midi(np.asarray(analysis["f0_hz"]))
        in_db = np.asarray(analysis.get("loudness_db") or np.zeros_like(in_t))
    dev_factor = TUNING_DEVIATION.get(tuning, 0.35)

    targets = []
    for i, (n, (st, en)) in enumerate(zip(notes, times)):
        m = float(n["midi"])
        if tuning == "hard":
            m = float(snap_to_scale(m, pcs)) if pcs else float(round(m))
        elif tuning == "natural":
            m = float(round(m))
        m += transpose + float(n.get("detune_cents", 0.0)) / 100.0
        targets.append(m)
        a, b = int(st * FPS), min(n_frames, int(np.ceil(en * FPS)))
        midi[a:b] = m
        owner[a:b] = i
        vel = float(n.get("velocity", 0.8)) * float(n.get("fx", {}).get("vel_mult", 1.0))
        env = np.ones(b - a)
        att, rel = int(0.03 * FPS), int(0.05 * FPS)
        if len(env) > att + rel:
            env[:att] = np.linspace(0.2, 1, att)
            env[-rel:] = np.linspace(1, 0.25, rel)
        amp[a:b] = np.maximum(amp[a:b], vel * env)

        # deviation from the detected contour (only for untouched, detected notes)
        if in_t is not None and not n.get("edited") and not n.get("generated") and dev_factor > 0:
            src = np.interp(t[a:b] - (st - n["start"]), in_t, np.nan_to_num(in_m, nan=-1))
            ok = src > 0
            if ok.sum() > 3:
                centre = np.median(src[ok])
                dev = np.clip(src - centre, -1.5, 1.5) * dev_factor
                if tuning == "original":
                    dev += centre - float(n["midi"])  # keep the singer's own intonation
                dev[~ok] = 0
                midi[a:b] += gaussian_filter1d(dev, 1.0)
                db = np.interp(t[a:b], in_t, in_db)
                rel_db = np.clip(db - np.median(db[ok]), -12, 6)
                amp[a:b] *= 10 ** (rel_db * 0.7 / 20)

    # fill short gaps so pitch is continuous for legato / transitions
    last = np.nan
    gap_len = 0
    for k in range(n_frames):
        if np.isnan(midi[k]):
            gap_len += 1
        else:
            if gap_len and gap_len < 0.15 * FPS and not np.isnan(last):
                midi[k - gap_len:k] = last
            gap_len = 0
            last = midi[k]
    first_valid = np.argmax(~np.isnan(midi)) if np.any(~np.isnan(midi)) else 0
    midi[:first_valid] = midi[first_valid] if np.any(~np.isnan(midi)) else 60
    for k in range(1, n_frames):
        if np.isnan(midi[k]):
            midi[k] = midi[k - 1]

    # transitions, scoops, slides, vibrato, gamaka
    for i, (n, (st, en)) in enumerate(zip(notes, times)):
        fx = n.get("fx") or {}
        a, b = int(st * FPS), min(n_frames, int(np.ceil(en * FPS)))
        if i > 0 and times[i - 1][1] > st - 0.15:
            prev = targets[i - 1]
            T = 0.008 if tuning == "hard" else 0.035
            T = max(T, float(fx.get("slide_ms", 0)) / 1000.0)
            if n.get("slide_in") and tuning != "hard":
                T = max(T, 0.07)
            s0, s1 = int((st - 0.6 * T) * FPS), int((st + 0.4 * T) * FPS)
            s0 = max(s0, 0)
            if s1 > s0:
                w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, s1 - s0))
                base = midi[s0:s1].copy()
                midi[s0:s1] = prev * (1 - w) + base * w
        sc = fx.get("scoop")
        if sc:
            L = int(sc["ms"] / 1000 * FPS)
            if L > 1:
                curve = sc["cents"] / 100.0 * (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, L)))
                midi[a:a + L] += curve[: max(0, min(L, n_frames - a))]
        vib = fx.get("vibrato")
        if vib:
            s = a + int(vib["delay"] * FPS)
            if b > s:
                tt = np.arange(b - s) / FPS
                fade = np.clip(tt / 0.18, 0, 1)
                midi[s:b] += vib["depth"] / 100.0 * fade * np.sin(2 * np.pi * vib["rate"] * tt)
        gm = fx.get("gamaka")
        if gm:
            s = a + int(gm["delay"] * FPS)
            if b > s:
                tt = np.arange(b - s) / FPS
                midi[s:b] += gm["interval"] / 100.0 * 0.5 * (1 - np.cos(2 * np.pi * gm["rate"] * tt))

    # natural micro-wander (seeded; part of voice timbre)
    r = np.random.default_rng([seed, 4242])
    wander = gaussian_filter1d(r.normal(0, 1, n_frames), 12)
    wander = wander / (np.std(wander) + 1e-9) * voice.get("jitter_cents", 5) / 100.0 * 0.6
    if tuning == "hard":
        wander *= 0.1
    midi = midi + wander
    amp = gaussian_filter1d(amp, 1.0)

    phones = build_phone_timeline(notes, times)
    # onset consonants sit before the note start (so the vowel lands on the beat): give them level
    vel = {n["id"]: float(n.get("velocity", 0.8)) * float(n.get("fx", {}).get("vel_mult", 1.0)) for n in notes}
    for sg in phones:
        if sg["role"] == "onset":
            a, b = int(sg["start"] * FPS), int(np.ceil(sg["end"] * FPS)) + 1
            amp[max(a, 0):b] = np.maximum(amp[max(a, 0):b], 0.85 * vel[sg["note"]])
    plan = {"fps": FPS, "n_frames": n_frames, "duration": n_frames / FPS, "f0_hz": midi_to_hz(midi), "midi": midi,
            "amp": amp, "phones": phones, "voice": voice, "timbre": perf.get("timbre", {}),
            "timbre_offset": perf.get("timbre_offset", {}), "notes": notes, "note_times": times}
    if window:
        plan["window"] = window
    return plan


def build_phone_timeline(notes, times):
    segs = []
    for i, (n, (st, en)) in enumerate(zip(notes, times)):
        ph = n.get("phonemes") or {"onset": [], "vowel": "a", "coda": []}
        vowel = ph.get("vowel") or "a"
        onset = split_phones(ph.get("onset", [])) if not n.get("continuation") else []
        coda = split_phones(ph.get("coda", []))
        if vowel == "m":  # hum
            vowel, coda = "u", ["m"] + coda
        prev_end = times[i - 1][1] if i else 0.0
        room_before = max(0.0, st - (times[i - 1][0] + 0.6 * (times[i - 1][1] - times[i - 1][0]))) if i else st
        on_d = [lookup(c).get("dur", 0.06) for c in onset]
        tot = sum(on_d)
        if tot > 0:
            scale = min(1.0, room_before / tot) if room_before > 0 else 0.5
            scale = max(scale, 0.45)
            cur = st - tot * scale
            for c, d in zip(onset, on_d):
                segs.append({"ph": c, "start": cur, "end": cur + d * scale, "role": "onset", "note": n["id"]})
                cur += d * scale
        co_d = [lookup(c).get("dur", 0.06) for c in coda]
        ctot = sum(co_d)
        cscale = min(1.0, 0.4 * (en - st) / ctot) if ctot else 1.0
        v_end = en - ctot * cscale
        if vowel in DIPHTHONGS:
            v1, v2 = DIPHTHONGS[vowel]
            mid = st + 0.62 * (v_end - st)
            segs.append({"ph": v1, "start": st, "end": mid, "role": "vowel", "note": n["id"]})
            segs.append({"ph": v2, "start": mid, "end": v_end, "role": "vowel", "note": n["id"]})
        else:
            segs.append({"ph": vowel, "start": st, "end": v_end, "role": "vowel", "note": n["id"]})
        cur = v_end
        for c, d in zip(coda, co_d):
            segs.append({"ph": c, "start": cur, "end": cur + d * cscale, "role": "coda", "note": n["id"]})
            cur += d * cscale
    return segs
