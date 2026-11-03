import io
import time
import zipfile

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from mixes import song
from synth import hum
from kuralkoodu.export import read_wav_header
from kuralkoodu.server import app

c = TestClient(app)


def wait(job, limit=1800):
    for _ in range(limit * 20):
        j = c.get(f"/api/jobs/{job['id']}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.05)
    raise TimeoutError


def upload(pid, y, sr, name="song.wav"):
    buf = io.BytesIO(); sf.write(buf, y, sr, format="WAV", subtype="FLOAT"); buf.seek(0)
    r = c.post(f"/api/projects/{pid}/audio", files={"file": (name, buf, "audio/wav")}, data={"role": "input"})
    assert r.status_code == 200, r.text


def test_full_song_separate_and_export_vocals_alone():
    pid = c.post("/api/projects", json={"name": "Duet Mix"}).json()["id"]
    mix, v, b = song(20, seed=2)
    upload(pid, mix, 44100)
    c.put(f"/api/projects/{pid}", json={"input_type": "song", "lyrics": {"text": "கண்ணே கலைமானே " * 6, "language": "ta", "dialect": "standard", "glides": True}})
    j = wait(c.post(f"/api/projects/{pid}/analyze").json())     # song mode separates first
    assert j["status"] == "done", j
    proj = c.get(f"/api/projects/{pid}").json()
    assert proj["separation"]["method"] == "builtin" and proj["separation"]["channels"] == 2
    a = c.get(f"/api/projects/{pid}/analysis").json()
    assert a["source"] == "separated_decoded.wav"
    # waveform peaks + playback for the new lanes
    for src in ("input", "vocals", "instrumental"):
        assert c.get(f"/api/projects/{pid}/play/{src}").status_code == 200
        pk = c.get(f"/api/projects/{pid}/peaks/{src}?n=500").json()
        assert abs(pk["duration"] - 20) < 0.01 and len(pk["max"]) >= 400
    # export vocals alone -> a single WAV
    j = wait(c.post(f"/api/projects/{pid}/separation/export", json={"stems": ["vocals"], "sample_rate": 48000, "bit_depth": "24"}).json())
    assert j["status"] == "done", j
    name = j["result"]["file"]
    assert name == "DuetMix_VocalsIsolated_48k24b.wav"
    h = read_wav_header(c.get(f"/api/projects/{pid}/exports/{name}").content)
    assert h["sample_rate"] == 48000 and h["bits"] == 24 and h["channels"] == 2 and h["frames"] == 20 * 48000
    # vocals + instrumental -> zip
    j = wait(c.post(f"/api/projects/{pid}/separation/export", json={"stems": ["vocals", "instrumental"], "bit_depth": "32f"}).json())
    z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/projects/{pid}/exports/{j['result']['file']}").content))
    wavs = [n for n in z.namelist() if n.endswith(".wav")]
    assert len(wavs) == 2
    # render + Logic export with the instrumental stem included, all aligned
    j = wait(c.post(f"/api/projects/{pid}/render", json={"mode": "full"}).json())
    assert j["status"] == "done", j
    j = wait(c.post(f"/api/projects/{pid}/export", json={"include_instrumental": True}).json())
    assert j["status"] == "done", j
    z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/projects/{pid}/exports/{j['result']['zip']}").content))
    hs = {n: read_wav_header(z.read(n)) for n in z.namelist() if n.endswith(".wav")}
    assert any("_Instrumental_" in n for n in hs) and len({h["frames"] for h in hs.values()}) == 1


def test_four_minute_song_renders_in_full():
    pid = c.post("/api/projects", json={"name": "Long One"}).json()["id"]
    rng = np.random.default_rng(9)
    y, sr, _ = hum(list(60 + rng.choice([0, 2, 4, 5, 7, 9], size=490)), bpm=122)
    assert len(y) / sr >= 240
    upload(pid, y, sr, "long.wav")
    c.put(f"/api/projects/{pid}", json={"lyrics": {"text": "நிலா நிலா ஓடி வா நில்லாமல் ஓடி வா " * 40, "language": "ta", "dialect": "standard", "glides": True}})
    t0 = time.time()
    j = wait(c.post(f"/api/projects/{pid}/analyze").json())
    assert j["status"] == "done", j
    j = wait(c.post(f"/api/projects/{pid}/render", json={"mode": "full"}).json())
    assert j["status"] == "done", j
    meta = c.get(f"/api/projects/{pid}/renders/full").json()
    assert meta["duration_s"] >= 240
    assert time.time() - t0 < 300
