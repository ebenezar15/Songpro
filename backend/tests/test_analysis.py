import numpy as np
import pytest

from synth import hum
from vocalforge.analysis import analyze

MEL = [60, 62, 64, 65, 67, 65, 64, 62, 60, 64, 67, 72, 71, 67, 64, 62] * 2


@pytest.mark.parametrize("bpm", [90.0, 120.0, 137.0])
def test_tempo_and_pitch_accuracy(bpm):
    y, sr, onsets = hum(MEL, bpm=bpm)
    a = analyze(y[:, None], sr)
    # tempo error < 1 BPM
    assert abs(a.tempo["bpm"] - bpm) < 1.0, a.tempo
    # notes: one per hummed note
    assert len(a.notes) == len(MEL), [round(n["midi"], 2) for n in a.notes]
    # median pitch error < 25 cents
    err = [abs(n["midi"] - m) * 100 for n, m in zip(a.notes, MEL)]
    assert np.median(err) < 25, err
    # onset error
    on_err = [abs(n["start"] - o) for n, o in zip(a.notes, onsets)]
    assert np.median(on_err) < 0.03


def test_key_detection():
    y, sr, _ = hum(MEL, bpm=120)
    a = analyze(y[:, None], sr)
    assert a.key["tonic"] == "C" and a.key["mode"] == "major", a.key


def test_dynamics_and_vibrato():
    loud = [-6, -24] * 8
    y, sr, _ = hum([64] * 2 + [67] * 2 + [69] * 12, bpm=60, beats_per_note=1, loud_db=loud, vibrato=60)
    a = analyze(y[:, None], sr)
    vel = [n["velocity"] for n in a.notes]
    assert vel[0] > vel[1]
    assert any(n["vibrato"] for n in a.notes)


def test_silence_and_short_fail_loudly():
    with pytest.raises(ValueError):
        analyze(np.zeros((44100 * 2, 1), np.float32), 44100)
    with pytest.raises(ValueError):
        analyze(np.zeros((1000, 1), np.float32), 44100)
