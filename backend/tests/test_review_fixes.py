"""Regression tests for the code-review fixes."""
import io
import threading
import time

import soundfile as sf
from fastapi.testclient import TestClient

from mixes import song
from synth import hum
from kuralkoodu.jobs import JobManager
from kuralkoodu.server import app

c = TestClient(app)


def wait(job):
    for _ in range(6000):
        j = c.get(f"/api/jobs/{job['id']}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.05)
    raise TimeoutError


def project_with_hum(name="Fix"):
    pid = c.post("/api/projects", json={"name": name}).json()["id"]
    y, sr, _ = hum([60, 62, 64, 65, 67, 65, 64, 62], bpm=100)
    buf = io.BytesIO(); sf.write(buf, y, sr, format="WAV"); buf.seek(0)
    c.post(f"/api/projects/{pid}/audio", files={"file": ("h.wav", buf, "audio/wav")}, data={"role": "input"})
    return pid


def test_foreign_host_and_origin_are_rejected():
    assert c.get("/api/health", headers={"host": "evil.example:8765"}).status_code == 403
    r = c.post("/api/projects", json={"name": "x"}, headers={"origin": "https://evil.example"})
    assert r.status_code == 403
    assert c.post("/api/projects", json={"name": "x"}, headers={"origin": "http://127.0.0.1:5173"}).status_code == 200


def test_invalid_settings_are_rejected_before_rendering():
    pid = project_with_hum()
    assert c.put(f"/api/projects/{pid}", json={"voice": {"gender": "robot", "age": "adult"}}).status_code == 400
    assert c.put(f"/api/projects/{pid}", json={"transpose": "high"}).status_code == 400
    assert c.put(f"/api/projects/{pid}", json={"flavour": {"tuning": "auto"}}).status_code == 400
    wait(c.post(f"/api/projects/{pid}/analyze").json())
    perf = c.get(f"/api/projects/{pid}").json()["performance"]
    assert c.put(f"/api/projects/{pid}/performance", json={"tempo": {**perf["tempo"], "bpm": 0}}).status_code == 400
    bad = [{**perf["notes"][0], "dur": -1}]
    assert c.put(f"/api/projects/{pid}/performance", json={"notes": bad}).status_code == 400
    assert c.get(f"/api/projects/{pid}/flavour").status_code == 200


def test_unchanged_lyrics_keep_note_edits():
    pid = project_with_hum()
    L = {"text": "நிலா நிலா ஓடி வா", "language": "ta", "dialect": "standard", "glides": True}
    c.put(f"/api/projects/{pid}", json={"lyrics": L})
    wait(c.post(f"/api/projects/{pid}/analyze").json())
    notes = c.get(f"/api/projects/{pid}").json()["performance"]["notes"]
    notes[1] = {**notes[1], "lyric": "வா", "lyric_dirty": True, "edited": True}
    c.put(f"/api/projects/{pid}/performance", json={"notes": notes})
    c.put(f"/api/projects/{pid}", json={"lyrics": L})  # same lyrics again (e.g. textarea blur)
    assert c.get(f"/api/projects/{pid}").json()["performance"]["notes"][1]["lyric"] == "வா"


def test_new_input_clears_old_separation():
    pid = c.post("/api/projects", json={"name": "Sep"}).json()["id"]
    mix, _, _ = song(8, seed=1)
    buf = io.BytesIO(); sf.write(buf, mix, 44100, format="WAV"); buf.seek(0)
    c.post(f"/api/projects/{pid}/audio", files={"file": ("s.wav", buf, "audio/wav")}, data={"role": "input"})
    assert wait(c.post(f"/api/projects/{pid}/separate", json={"method": "builtin"}).json())["status"] == "done"
    assert c.get(f"/api/projects/{pid}").json()["separation"]
    buf.seek(0)
    c.post(f"/api/projects/{pid}/audio", files={"file": ("s2.wav", buf, "audio/wav")}, data={"role": "input"})
    assert c.get(f"/api/projects/{pid}").json()["separation"] is None


def test_jobs_for_one_song_run_one_at_a_time_and_history_is_pruned():
    jm = JobManager(workers=2)
    active, peak = [0], [0]
    lock = threading.Lock()

    def work(job):
        with lock:
            active[0] += 1; peak[0] = max(peak[0], active[0])
        time.sleep(0.2)
        with lock:
            active[0] -= 1
        return {}

    a, b = jm.submit("x", "p1", work), jm.submit("x", "p1", work)
    for _ in range(100):
        if a.status == b.status == "done":
            break
        time.sleep(0.05)
    assert a.status == b.status == "done" and peak[0] == 1
    for _ in range(250):
        jm.submit("y", None, lambda job: {})
    time.sleep(0.5)
    jm.submit("y", None, lambda job: {})
    assert len(jm.jobs) <= 202
