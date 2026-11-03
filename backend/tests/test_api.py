import io
import time
import zipfile

import soundfile as sf
from fastapi.testclient import TestClient

from synth import hum
from kuralkoodu.server import app

c = TestClient(app)


def wait(job):
    for _ in range(600):
        j = c.get(f"/api/jobs/{job['id']}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.05)
    raise TimeoutError


def test_end_to_end():
    pid = c.post("/api/projects", json={"name": "Test Song"}).json()["id"]
    y, sr, _ = hum([60, 62, 64, 65, 67, 65, 64, 62] * 2, bpm=100)
    buf = io.BytesIO(); sf.write(buf, y, sr, format="WAV"); buf.seek(0)
    r = c.post(f"/api/projects/{pid}/audio", files={"file": ("hum.wav", buf, "audio/wav")}, data={"role": "input"})
    assert r.status_code == 200
    # reference upload requires consent
    buf.seek(0)
    r = c.post(f"/api/projects/{pid}/audio", files={"file": ("ref.wav", buf, "audio/wav")}, data={"role": "reference"})
    assert r.status_code == 400
    c.put(f"/api/projects/{pid}", json={"lyrics": {"text": "நிலா நிலா ஓடி வா நில்லாமல் ஓடி வா", "language": "ta", "dialect": "standard", "glides": True}})
    j = wait(c.post(f"/api/projects/{pid}/analyze").json())
    assert j["status"] == "done", j
    # cached second time
    j2 = wait(c.post(f"/api/projects/{pid}/analyze").json())
    assert j2["result"]["cached"] is True
    proj = c.get(f"/api/projects/{pid}").json()
    assert proj["performance"]["notes"][0]["lyric"] == "நி"
    # user edit overrides detection
    notes = proj["performance"]["notes"]
    notes[0]["midi"] = 59; notes[0]["edited"] = True
    c.put(f"/api/projects/{pid}/performance", json={"notes": notes})
    fl = c.get(f"/api/projects/{pid}/flavour").json()
    assert fl["notes"][0]["midi"] == 59
    j = wait(c.post(f"/api/projects/{pid}/render", json={"mode": "preview", "bars": 2}).json())
    assert j["status"] == "done", j
    c.put(f"/api/projects/{pid}", json={"flavour": {**proj["flavour"], "harmonies": {"enabled": True, "intervals": [2]}, "doubles": {"enabled": True, "count": 1}}})
    j = wait(c.post(f"/api/projects/{pid}/render", json={"mode": "full"}).json())
    assert j["status"] == "done", j
    assert set(j["result"]["stems"]) == {"lead", "doubles", "harmonies"}
    assert c.get(f"/api/projects/{pid}/renders/full/lead.wav").status_code == 200
    j = wait(c.post(f"/api/projects/{pid}/export", json={"sample_rate": 48000, "bit_depth": "24"}).json())
    assert j["status"] == "done", j
    z = c.get(f"/api/projects/{pid}/exports/{j['result']['zip']}")
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert any(n.startswith("TestSong_LeadDry_Female_YoungAdult_100bpm_") for n in names), names
    # commercial mode blocks unclear licences (seed-vc) but allows built-in
    c.put(f"/api/projects/{pid}", json={"commercial_mode": True})
    j = wait(c.post(f"/api/projects/{pid}/export", json={}).json())
    assert j["status"] == "done"


def test_unavailable_engine_reports_reason():
    pid = c.post("/api/projects", json={"name": "x"}).json()["id"]
    y, sr, _ = hum([60, 62, 64, 65], bpm=100)
    buf = io.BytesIO(); sf.write(buf, y, sr, format="WAV"); buf.seek(0)
    c.post(f"/api/projects/{pid}/audio", files={"file": ("hum.wav", buf, "audio/wav")}, data={"role": "input"})
    wait(c.post(f"/api/projects/{pid}/analyze").json())
    c.put(f"/api/projects/{pid}", json={"voice": {"gender": "female", "age": "adult", "voice_id": None, "engine": "soulx-singer"}})
    j = wait(c.post(f"/api/projects/{pid}/render", json={"mode": "preview"}).json())
    assert j["status"] == "error" and "SoulX" in j["error"]


def test_bad_upload():
    pid = c.post("/api/projects", json={"name": "x"}).json()["id"]
    r = c.post(f"/api/projects/{pid}/audio", files={"file": ("x.wav", io.BytesIO(b"garbage"), "audio/wav")}, data={"role": "input"})
    assert r.status_code == 400
