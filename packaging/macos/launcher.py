"""Kuralkoodu desktop launcher.

Starts the local engine (FastAPI on 127.0.0.1, a free port) inside the app and shows the
studio in a native macOS window (WKWebView via pywebview). If the native window cannot be
created, the studio opens in the default browser instead.

    Kuralkoodu --self-test   start the engine, analyse + render + export a generated hum, exit 0/1
    Kuralkoodu --browser     use the default browser instead of the app window
"""
from __future__ import annotations

import io
import json
import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _log_file() -> Path:
    d = Path(os.environ.get("KURALKOODU_HOME", Path.home() / "Kuralkoodu")) / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / "kuralkoodu.log"


def start_server(port: int):
    import uvicorn

    from kuralkoodu.server import app
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True, name="kuralkoodu-engine")
    t.start()
    for _ in range(300):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as r:
                if r.status == 200:
                    return server
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("The Kuralkoodu engine did not start.")


def _req(method, url, body=None, files=None):
    import uuid
    headers = {}
    data = None
    if files:
        boundary = uuid.uuid4().hex
        parts = []
        for k, v in (body or {}).items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        for k, (fn, content) in files.items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{fn}"\r\nContent-Type: audio/wav\r\n\r\n'.encode() + content + b"\r\n")
        data = b"".join(parts) + f"--{boundary}--\r\n".encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = r.read()
        return json.loads(raw) if r.headers.get_content_type() == "application/json" else raw


def self_test(port: int) -> int:
    """End-to-end check of the packaged app: import, analyse, render, separate, export."""
    import numpy as np
    import soundfile as sf
    base = f"http://127.0.0.1:{port}"

    def wait(job):
        while job["status"] in ("queued", "running"):
            time.sleep(0.2)
            job = _req("GET", f"{base}/api/jobs/{job['id']}")
        if job["status"] != "done":
            raise RuntimeError(f"{job['kind']} failed: {job.get('error')}")
        return job

    sr = 44100
    t = np.arange(int(sr * 6)) / sr
    mel = [60, 62, 64, 65, 67, 65, 64, 62, 60, 64, 67, 72]
    y = np.zeros_like(t)
    for i, m in enumerate(mel):
        s, e = int((0.3 + i * 0.45) * sr), int((0.3 + i * 0.45 + 0.4) * sr)
        f = 440 * 2 ** ((m - 69) / 12)
        y[s:e] = sum(np.sin(2 * np.pi * f * k * t[s:e]) / k for k in range(1, 6)) * 0.2
    stereo = np.stack([y, y * 0.9], axis=1).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV")
    pid = _req("POST", f"{base}/api/projects", {"name": "Self Test"})["id"]
    _req("POST", f"{base}/api/projects/{pid}/audio", {"role": "input"}, files={"file": ("selftest.wav", buf.getvalue())})
    _req("PUT", f"{base}/api/projects/{pid}", {"lyrics": {"text": "நிலா நிலா ஓடி வா நில்லாமல் ஓடி வா", "language": "ta", "dialect": "standard", "glides": True}})
    wait(_req("POST", f"{base}/api/projects/{pid}/analyze"))
    wait(_req("POST", f"{base}/api/projects/{pid}/render", {"mode": "full"}))
    wait(_req("POST", f"{base}/api/projects/{pid}/separate", {"method": "builtin"}))
    sep = wait(_req("POST", f"{base}/api/projects/{pid}/separation/export", {"stems": ["vocals"]}))
    ex = wait(_req("POST", f"{base}/api/projects/{pid}/export", {"sample_rate": 48000, "bit_depth": "24"}))
    ui = urllib.request.urlopen(f"{base}/", timeout=10).read()
    assert b"<div id=\"root\">" in ui, "UI files are missing from the bundle"
    print(json.dumps({"ok": True, "separation_export": sep["result"]["file"], "logic_export": ex["result"]["zip"],
                      "frames": ex["result"]["report"]["frames"]}))
    return 0


def main():
    args = sys.argv[1:]
    port = int(os.environ.get("KURALKOODU_PORT") or _free_port())
    try:
        start_server(port)
    except Exception as e:  # never fail silently: show why
        _log_file().write_text(f"{time.ctime()} engine failed: {e!r}\n")
        raise
    url = f"http://127.0.0.1:{port}/"
    if "--self-test" in args:
        sys.exit(self_test(port))
    if "--browser" not in args:
        try:
            import webview  # pywebview: WKWebView on macOS
            webview.settings["ALLOW_DOWNLOADS"] = True
            webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
            webview.create_window("Kuralkoodu", url, width=1480, height=940, min_size=(980, 640))
            webview.start()
            return
        except Exception as e:
            with open(_log_file(), "a") as f:
                f.write(f"{time.ctime()} native window unavailable, using the browser: {e!r}\n")
    import webbrowser
    webbrowser.open(url)
    print(f"Kuralkoodu is running at {url} (close this window or press Ctrl+C to quit)")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
