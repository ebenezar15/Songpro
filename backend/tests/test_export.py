import io
import json
import zipfile

import numpy as np
import soundfile as sf

from kuralkoodu.export import build_export, midi_bytes, read_wav_header, stem_filename, wav_bytes


def test_wav_headers():
    x = np.sin(np.linspace(0, 100, 48000)).astype(np.float32) * 0.5
    for bd, tag, bits in (("24", 1, 24), ("32f", 3, 32)):
        b = wav_bytes(x, 48000, bd)
        h = read_wav_header(b)
        assert h["format_tag"] == tag and h["bits"] == bits and h["sample_rate"] == 48000 and h["frames"] == 48000
        y, sr = sf.read(io.BytesIO(b))
        assert np.max(np.abs(y - x)) < 1e-6 if bd == "32f" else np.max(np.abs(y - x)) < 1e-6 * 2
    b = wav_bytes(np.stack([x, x], 1), 96000, "24", bwf=True, time_reference=123456)
    h = read_wav_header(b)
    assert h["channels"] == 2 and h["time_reference"] == 123456 and h["chunks"][0] == "bext"


def test_filename():
    assert stem_filename("My Song", "lead", "female", "young_adult", 120, "Cmin", 48000, "24") == \
        "MySong_LeadDry_Female_YoungAdult_120bpm_Cmin_48k24b.wav"
    assert stem_filename("x", "doubles", "male", "senior", 92.4, "Amaj", 44100, "32f").endswith("_92bpm_Amaj_44.1k32f.wav")


def test_export_alignment_and_gain():
    sr = 48000
    lead = np.zeros(sr * 2, np.float32); lead[1000] = 1.6   # would clip
    harm = np.zeros(sr * 1, np.float32); harm[500] = 0.5     # shorter stem
    session = {"voice": {"gender": "female", "age": "adult"}, "tempo": {"bpm": 100}, "seed": 1}
    notes = [{"start": 0.5, "dur": 0.5, "midi": 60, "velocity": 0.8, "lyric": "நி"}]
    data, rep = build_export({"lead": lead, "harmonies": harm}, sr, {"sample_rate": 44100, "bit_depth": "24", "bwf": True, "processed": True},
                             session, notes, [{"time": 0, "bpm": 100}], {"tonic": "A", "tonic_pc": 9, "mode": "minor", "short": "Amin"})
    z = zipfile.ZipFile(io.BytesIO(data))
    wavs = [n for n in z.namelist() if n.endswith(".wav")]
    assert len(wavs) == 3
    frames = {read_wav_header(z.read(n))["frames"] for n in wavs}
    assert len(frames) == 1 and frames.pop() == rep["frames"]          # sample-aligned, same length
    assert rep["gain_applied_db"] < 0 and not rep["normalised"]
    for n in wavs:
        y, _ = sf.read(io.BytesIO(z.read(n)))
        assert np.max(np.abs(y)) <= 10 ** (-1 / 20) + 1e-3          # no clipping
    sess = json.loads(z.read(next(n for n in z.namelist() if n.endswith("Session.json"))))
    assert sess["seed"] == 1 and sess["export"]["sample_rate"] == 44100
    mid = z.read(next(n for n in z.namelist() if n.endswith(".mid")))
    assert mid[:4] == b"MThd" and mid.count(b"MTrk") == 2


def test_midi_tempo_map():
    m = midi_bytes([{"start": 0, "dur": 1, "midi": 60}], [{"time": 0, "bpm": 120}, {"time": 4, "bpm": 90}])
    assert m.count(b"\xff\x51\x03") == 2
