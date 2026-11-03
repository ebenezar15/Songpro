import hashlib

import numpy as np

from synth import hum
from vocalforge.analysis import analyze
from vocalforge.engines import FormantSingerEngine, MockEngine
from vocalforge.flavour import apply_flavours
from vocalforge.lyrics import align_to_notes, parse_lyrics
from vocalforge.planner import build_plan
from vocalforge.service import performance_from_analysis
from vocalforge.voices import range_check, voice_params

MEL = [64, 64, 67, 69, 67, 64, 62, 64, 60, 62, 64, 67, 64, 62, 60, 60]


def _perf():
    y, sr, _ = hum(MEL, bpm=110, beats_per_note=1)
    a = analyze(y[:, None], sr).to_dict()
    perf = performance_from_analysis(a)
    perf["notes"], _ = align_to_notes(perf["notes"], parse_lyrics("கண்ணே கலைமானே கன்னி மயிலென கண்டேன் உன்னை நானே", "ta"))
    return perf, a


def test_creative_zero_is_faithful():
    perf, _ = _perf()
    p2, changes = apply_flavours(perf, {"creative": 0, "preset": "R&B", "vibrato": 100, "melisma": 100, "timing": 100})
    assert not changes
    for a, b in zip(perf["notes"], p2["notes"]):
        assert a["start"] == b["start"] and a["dur"] == b["dur"] and a["midi"] == b["midi"]
        assert b["fx"]["timing_ms"] == 0 and b["fx"]["vel_mult"] == 1.0


def test_flavours_are_seeded_and_undo_is_isolated():
    perf, _ = _perf()
    s = {"creative": 100, "vibrato": 60, "scoops": 60, "slides": 60, "melisma": 80, "timing": 50, "dynamics": 50, "seed": 7}
    p1, c1 = apply_flavours(perf, s)
    p2, c2 = apply_flavours(perf, s)
    assert p1 == p2 and c1 == c2
    victim = c1[0]["id"]
    p3, c3 = apply_flavours(perf, {**s, "disabled_changes": [victim]})
    others = [c for c in c3 if c["id"] != victim]
    assert others == [c for c in c1 if c["id"] != victim]
    assert next(c for c in c3 if c["id"] == victim)["enabled"] is False
    p4, _ = apply_flavours(perf, {**s, "seed": 8})
    assert p4 != p1


def test_render_is_deterministic_and_follows_performance():
    perf, a = _perf()
    v = voice_params("female", "young_adult")
    p, _ = apply_flavours(perf, {"creative": 0, "tuning": "natural"})
    eng = FormantSingerEngine()
    y1, sr = eng.render(build_plan(p, v, a, seed=3), 3)
    y2, _ = eng.render(build_plan(p, v, a, seed=3), 3)
    assert hashlib.sha256(y1.tobytes()).hexdigest() == hashlib.sha256(y2.tobytes()).hexdigest()
    # re-analyse the output: notes + timing stay close to the performance at Creative = 0
    r = analyze(y1[:, None], sr)
    ref = sorted(perf["notes"], key=lambda n: n["start"])
    got = r.notes
    assert abs(len(got) - len(ref)) <= 2, (len(got), len(ref))
    # match each ref note to the nearest detected onset
    errs_c, errs_t = [], []
    for n in ref:
        g = min(got, key=lambda x: abs(x["start"] - n["start"]))
        errs_t.append(abs(g["start"] - n["start"]))
        errs_c.append(abs(g["midi"] - round(n["midi"])) * 100)
    assert np.median(errs_c) < 25, errs_c
    assert np.median(errs_t) < 0.05, errs_t


def test_mock_engine_and_range_check():
    perf, a = _perf()
    v = voice_params("male", "adult")
    y, sr = MockEngine().render(build_plan(perf, v, a), 0)
    assert sr == 48000 and len(y) > sr * 5
    rc = range_check(perf["notes"], voice_params("male", "senior"))
    assert "suggestions" in rc
