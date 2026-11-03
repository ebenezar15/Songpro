"""Project storage and the analysis / render / export pipeline used by the API."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from . import __version__
from .analysis import analyze
from .audio_io import load_audio, to_mono, write_preview_wav
from .engines import Cancelled, EngineUnavailable, get_engine
from .export import build_export
from .flavour import DEFAULT_SETTINGS, PRESETS, apply_flavours, make_adlibs, make_double, make_harmony
from .licenses import REGISTRY, commercial_allowed
from .lyrics import align_to_notes, parse_lyrics
from .planner import build_plan
from .voices import analyse_reference, range_check, voice_params


def home() -> Path:
    p = Path(os.environ.get("KURALKOODU_HOME", Path.home() / "Kuralkoodu"))
    (p / "projects").mkdir(parents=True, exist_ok=True)
    return p


def pdir(pid: str) -> Path:
    if not pid or any(c in pid for c in "/\\.") or len(pid) > 40:
        raise ValueError("bad project id")
    return home() / "projects" / pid


DEFAULT_PROJECT = {
    "input": None, "input_type": "hum", "reference": None,
    "lyrics": {"text": "", "language": "ta", "dialect": "standard", "glides": True},
    "voice": {"gender": "female", "age": "young_adult", "voice_id": "meera", "engine": "formant"},
    "flavour": copy.deepcopy(DEFAULT_SETTINGS), "performance": None, "commercial_mode": False,
    "song_name": "Untitled", "transpose": 0,
}


def create_project(name: str) -> dict:
    pid = uuid.uuid4().hex[:10]
    d = pdir(pid)
    d.mkdir(parents=True)
    proj = {"id": pid, "name": name or "Untitled", "created": time.time(), **copy.deepcopy(DEFAULT_PROJECT), "song_name": name or "Untitled"}
    save_project(proj)
    return proj


def load_project(pid: str) -> dict:
    f = pdir(pid) / "project.json"
    if not f.exists():
        raise FileNotFoundError(pid)
    proj = json.loads(f.read_text())
    for k, v in DEFAULT_PROJECT.items():
        proj.setdefault(k, copy.deepcopy(v))
    return proj


def save_project(proj: dict):
    d = pdir(proj["id"])
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "project.json.tmp"
    tmp.write_text(json.dumps(proj, indent=1, ensure_ascii=False, default=float))
    tmp.replace(d / "project.json")


def list_projects():
    out = []
    for f in sorted((home() / "projects").glob("*/project.json"), key=lambda p: -p.stat().st_mtime):
        try:
            p = json.loads(f.read_text())
            out.append({"id": p["id"], "name": p["name"], "created": p.get("created"), "has_analysis": (f.parent / "analysis.json").exists()})
        except Exception:
            continue
    return out


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_play_copy(path: Path, x: np.ndarray, sr: int):
    """16-bit copy for in-app playback (half the size of float; long songs load faster in the browser)."""
    x = np.asarray(x, np.float32)
    pk = float(np.max(np.abs(x))) if x.size else 0.0
    sf.write(str(path), x / pk if pk > 1.0 else x, sr, subtype="PCM_16")


def peaks(path: Path, n: int = 2000) -> dict:
    """Min/max waveform overview (cached next to the file)."""
    cache = path.with_suffix(f".peaks{n}.json")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return json.loads(cache.read_text())
    info = sf.info(str(path))
    total = info.frames
    step = max(1, total // n)
    mins, maxs = [], []
    with sf.SoundFile(str(path)) as f:
        while True:
            blk = f.read(step * 64, dtype="float32", always_2d=True)
            if not len(blk):
                break
            m = blk.mean(axis=1)
            k = len(m) // step
            if k:
                seg = m[: k * step].reshape(k, step)
                mins += seg.min(axis=1).round(4).tolist()
                maxs += seg.max(axis=1).round(4).tolist()
    out = {"duration": total / info.samplerate, "min": mins, "max": maxs}
    cache.write_text(json.dumps(out))
    return out


def store_audio(pid: str, filename: str, data: bytes, role: str) -> dict:
    d = pdir(pid)
    ext = Path(filename).suffix.lower() or ".wav"
    dest = d / f"{role}{ext}"
    for old in d.glob(f"{role}.*"):
        old.unlink()
    dest.write_bytes(data)
    x, sr = load_audio(dest)  # validates decoding now (never fail silently later)
    # keep a decoded float copy for fast playback / processing
    write_preview_wav(d / f"{role}_decoded.wav", to_mono(x), sr)
    write_play_copy(d / f"{role}_play.wav", x, sr)
    if role == "input":  # new input invalidates old separation / renders
        for sub in ("separation", "renders"):
            if (d / sub).exists():
                shutil.rmtree(d / sub)
        for f in ("separated_decoded.wav", "analysis.json"):
            if (d / f).exists():
                (d / f).unlink()
    return {"file": dest.name, "original_name": filename, "sample_rate": sr, "channels": x.shape[1],
            "duration": round(len(x) / sr, 3), "hash": file_hash(dest)}


# ---------------------------------------------------------------- analysis
def run_analysis(job, pid: str, force=False, use_separated=False):
    proj = load_project(pid)
    if not proj.get("input"):
        raise ValueError("Import audio first.")
    d = pdir(pid)
    if proj.get("input_type") == "song":
        use_separated = True
        if not (d / "separated_decoded.wav").exists():
            job.update(0.01, "Full song: separating the vocal first")
            run_separation(job, pid, proj.get("separation_method") or "builtin", progress_scale=0.6)
    src = d / ("separated_decoded.wav" if use_separated and (d / "separated_decoded.wav").exists() else "input_decoded.wav")
    key = file_hash(src)
    cache = d / "analysis.json"
    if cache.exists() and not force:
        cached = json.loads(cache.read_text())
        if cached.get("source_hash") == key:
            job.update(1.0, "Loaded cached analysis (no re-analysis needed)")
            return {"cached": True}
    x, sr = load_audio(src)
    t0 = time.time()
    base = job.progress
    res = analyze(x, sr, progress=lambda f, m: (job.update(base + (0.95 - base) * f, m), _check(job)), separated=src.name == "separated_decoded.wav")
    out = res.to_dict()
    out["source_hash"] = key
    out["source"] = src.name
    out["runtime_s"] = round(time.time() - t0, 2)
    cache.write_text(json.dumps(out))
    proj = load_project(pid)
    proj["performance"] = performance_from_analysis(out)
    if proj.get("input_type") == "sung":
        try:
            proj["source_timbre"] = analyse_reference(x, sr)
        except ValueError:
            pass
    if proj["lyrics"]["text"].strip():
        apply_lyrics(proj)
    save_project(proj)
    return {"cached": False, "runtime_s": out["runtime_s"], "notes": len(out["notes"])}


def _check(job):
    if job.cancelled():
        raise Cancelled()


def load_analysis(pid: str) -> dict | None:
    f = pdir(pid) / "analysis.json"
    return json.loads(f.read_text()) if f.exists() else None


def performance_from_analysis(a: dict) -> dict:
    return {"notes": copy.deepcopy(a["notes"]), "tempo": copy.deepcopy(a["tempo"]), "tempo_map": copy.deepcopy(a["tempo_map"]),
            "key": copy.deepcopy(a["key"]), "phrases": copy.deepcopy(a["phrases"]), "edited": False}


def apply_lyrics(proj: dict):
    perf = proj.get("performance")
    if not perf:
        return {"syllables": [], "report": {}}
    L = proj["lyrics"]
    syl = parse_lyrics(L["text"], L.get("language", "ta"), L.get("dialect", "standard"), L.get("glides", True))
    notes, report = align_to_notes(sorted(perf["notes"], key=lambda n: n["start"]), syl)
    perf["notes"] = notes
    return {"syllables": syl, "report": report}


def renumber_phrases(perf: dict, min_gap=0.25):
    notes = sorted(perf["notes"], key=lambda n: n["start"])
    phrases, cur = [], []
    for n in notes:
        if cur and n["start"] - (cur[-1]["start"] + cur[-1]["dur"]) >= min_gap:
            phrases.append(cur)
            cur = []
        cur.append(n)
    if cur:
        phrases.append(cur)
    perf["phrases"] = [{"id": f"p{i}", "start": p[0]["start"], "end": p[-1]["start"] + p[-1]["dur"], "note_ids": [n["id"] for n in p]}
                       for i, p in enumerate(phrases)]
    return perf


# ------------------------------------------------------------------ render
def current_voice(proj):
    v = proj["voice"]
    ref = (proj.get("reference") or {}).get("timbre") if (proj.get("reference") or {}).get("use") else None
    return voice_params(v["gender"], v["age"], v.get("voice_id"), ref)


def flavour_preview(proj):
    perf = copy.deepcopy(proj["performance"])
    perf["transpose"] = proj.get("transpose", 0)
    p2, changes = apply_flavours(perf, proj["flavour"])
    return p2, changes


def _window_notes(perf, start, end):
    p = copy.deepcopy(perf)
    p["notes"] = [n for n in p["notes"] if n["start"] < end and n["start"] + n["dur"] > start]
    for n in p["notes"]:
        n["start"] = round(n["start"] - start, 4)
    return p


def render_project(job, pid: str, mode="preview", start_bar=0, bars=8):
    proj = load_project(pid)
    if not proj.get("performance"):
        raise ValueError("Analyse the input first.")
    analysis = load_analysis(pid)
    engine = get_engine(proj["voice"].get("engine", "formant"))
    ok, msg = engine.available()
    if not ok:
        raise EngineUnavailable(f"{engine.name}: {msg}")
    seed = int(proj["flavour"].get("seed", 1234))
    voice = current_voice(proj)
    perf, changes = flavour_preview(proj)
    lang = proj["lyrics"].get("language", "ta")
    bpm = float(perf["tempo"]["bpm"])
    bar_s = 4 * 60.0 / bpm
    offset = 0.0
    if mode == "preview":
        offset = max(0.0, float(perf["tempo"].get("beat_offset", 0.0)) + start_bar * bar_s - 0.15)
        perf = _window_notes(perf, offset, offset + bars * bar_s)
        an = _shift_analysis(analysis, offset, offset + bars * bar_s)
    else:
        an = analysis
    job.update(0.05, f"Rendering {mode} with {engine.name} (seed {seed})")

    parts = {"lead": perf}
    fl = proj["flavour"]
    if fl.get("doubles", {}).get("enabled"):
        parts["doubles"] = [make_double(perf, seed, i) for i in range(int(fl["doubles"].get("count", 1)))]
    if fl.get("harmonies", {}).get("enabled"):
        parts["harmonies"] = [make_harmony(perf, int(s)) for s in fl["harmonies"].get("intervals", [2])]
    if fl.get("adlibs", {}).get("enabled"):
        parts["adlibs"] = [make_adlibs(perf, seed, lang)]

    d = pdir(pid)
    source = None
    if engine.id in ("dsp-convert",):
        src_file = d / ("separated_decoded.wav" if (d / "separated_decoded.wav").exists() else "input_decoded.wav")
        x, sr = load_audio(src_file)
        x = to_mono(x)
        if mode == "preview":
            x = x[int(offset * sr): int((offset + bars * bar_s) * sr)]
        source = (x, sr)
    if engine.id == "dsp-convert" and len(parts) > 1:
        raise EngineUnavailable("Doubles/harmonies/ad-libs need a synthesis engine (Formant or SoulX); turn them off for Voice Character mode.")
    total = sum(len(v) if isinstance(v, list) else 1 for v in parts.values())
    done = 0
    stems = {}
    sr_out = engine.native_sr
    for name, pl in parts.items():
        variants = pl if isinstance(pl, list) else [pl]
        acc = None
        for var in variants:
            plan = build_plan(var, voice, an if name == "lead" else None, seed=seed)
            plan["transpose"] = proj.get("transpose", 0)
            if source is not None:
                plan["source_audio"] = source
                plan["source_formant_scale"] = (proj.get("source_timbre") or {}).get("formant_scale", 1.0)
            plan["source_wav"] = str(d / "input_decoded.wav")
            if (proj.get("reference") or {}).get("use"):
                plan["reference_wav"] = str(d / "reference_decoded.wav")
            base = 0.05 + 0.9 * done / total
            span = 0.9 / total
            y, sr_out = engine.render(plan, seed, progress=lambda f, m, b=base, s=span: job.update(b + s * f, f"{name}: {m}"),
                                      cancel=job.cancelled)
            gain = {"lead": 1.0, "doubles": 0.7, "harmonies": 0.6, "adlibs": 0.6}[name] / (len(variants) ** 0.5)
            y = y * gain
            acc = y if acc is None else _mix(acc, y)
            done += 1
        stems[name] = acc
    L = max(len(v) for v in stems.values())
    stems = {k: np.pad(v, (0, L - len(v))) for k, v in stems.items()}
    out = d / "renders" / mode
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    hashes = {}
    for k, v in stems.items():
        write_preview_wav(out / f"{k}.wav", v, sr_out)
        hashes[k] = hashlib.sha256(np.asarray(v, np.float32).tobytes()).hexdigest()[:16]
    for k, v in stems.items():
        write_play_copy(out / f"{k}_play.wav", v, sr_out)
    meta = {"mode": mode, "engine": engine.info(), "seed": seed, "voice": voice, "flavour": proj["flavour"],
            "changes": changes, "offset_s": round(offset, 4), "bars": bars if mode == "preview" else None,
            "sample_rate": sr_out, "duration_s": round(L / sr_out, 3), "audio_sha256": hashes,
            "transpose": proj.get("transpose", 0), "app_version": __version__, "time": time.time(),
            "range_check": range_check(perf["notes"], voice)}
    (out / "render.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=float))
    return {"mode": mode, "stems": list(stems.keys()), "duration_s": meta["duration_s"], "audio_sha256": hashes}


def _mix(a, b):
    L = max(len(a), len(b))
    return np.pad(a, (0, L - len(a))) + np.pad(b, (0, L - len(b)))


def _shift_analysis(a, start, end):
    if not a:
        return None
    t = np.asarray(a["f0_times"])
    m = (t >= start) & (t < end)
    return {"f0_times": list(t[m] - start), "f0_hz": list(np.asarray(a["f0_hz"])[m]),
            "loudness_db": list(np.asarray(a["loudness_db"])[m]), "duration": end - start}


# -------------------------------------------------------------- separation
def run_separation(job, pid: str, method: str = "builtin", strength: float = 1.0, progress_scale: float = 1.0):
    from .separation import separate
    proj = load_project(pid)
    if not proj.get("input"):
        raise ValueError("Import a song first.")
    d = pdir(pid)
    x, sr = load_audio(d / proj["input"]["file"])  # original channels (stereo helps)
    t0 = time.time()
    base = job.progress
    voc, inst, info = separate(x, sr, method, progress=lambda f, m: (job.update(base + progress_scale * 0.95 * f, m), _check(job)),
                               cancel=job.cancelled, strength=strength)
    out = d / "separation"
    out.mkdir(exist_ok=True)
    sf.write(str(out / "vocals.wav"), voc, sr, subtype="FLOAT")
    sf.write(str(out / "instrumental.wav"), inst, sr, subtype="FLOAT")
    write_play_copy(out / "vocals_play.wav", voc, sr)
    write_play_copy(out / "instrumental_play.wav", inst, sr)
    write_preview_wav(d / "separated_decoded.wav", to_mono(voc), sr)
    info.update(runtime_s=round(time.time() - t0, 2), duration_s=round(len(x) / sr, 3), strength=strength,
                vocal_peak_dbfs=round(20 * np.log10(np.max(np.abs(voc)) + 1e-12), 2),
                instrumental_peak_dbfs=round(20 * np.log10(np.max(np.abs(inst)) + 1e-12), 2), time=time.time())
    proj = load_project(pid)
    proj["separation"] = json.loads(json.dumps(info, default=float))
    save_project(proj)
    return info


def export_separation(job, pid: str, opts: dict):
    """Export the isolated vocal and/or instrumental. One stem -> .wav, several -> .zip."""
    import io
    import zipfile
    from .audio_io import resample
    from .export import _slug, wav_bytes
    proj = load_project(pid)
    d = pdir(pid) / "separation"
    if not (d / "vocals.wav").exists():
        raise ValueError("Separate the song first.")
    stems = opts.get("stems") or ["vocals"]
    sr = int(opts.get("sample_rate") or proj["separation"]["sample_rate"])
    bd = str(opts.get("bit_depth", "24"))
    chans = opts.get("channels", "original")
    song = _slug(proj.get("song_name") or proj["name"])
    srs = f"{sr / 1000:g}k"
    tag = "32f" if bd == "32f" else f"{bd}b"
    label = {"vocals": "VocalsIsolated", "instrumental": "Instrumental"}
    data = {}
    for st in stems:
        y, ssr = sf.read(str(d / f"{st}.wav"), dtype="float32", always_2d=True)
        y = resample(y, ssr, sr)
        if chans == 1:
            y = y.mean(axis=1, keepdims=True)
        elif chans == 2 and y.shape[1] == 1:
            y = np.repeat(y, 2, axis=1)
        data[st] = y
        job.update(0.2 + 0.3 * len(data) / len(stems), f"Prepared {st}")
    # never normalise; only if 24-bit would clip, apply one common gain to every stem
    peak = max(float(np.max(np.abs(v))) for v in data.values())
    gain = 1.0 if bd == "32f" or peak <= 1.0 else 1.0 / peak
    files = {}
    for st, y in data.items():
        files[f"{song}_{label[st]}_{srs}{tag}.wav"] = wav_bytes(y * gain, sr, bd)
    ex = pdir(pid) / "exports"
    ex.mkdir(exist_ok=True)
    report = {"stems": stems, "sample_rate": sr, "bit_depth": bd, "gain_applied_db": round(20 * np.log10(gain), 2),
              "method": proj["separation"].get("method"), "frames": len(next(iter(data.values())))}
    if len(files) == 1:
        name, b = next(iter(files.items()))
        (ex / name).write_bytes(b)
        return {"file": name, "report": report}
    name = f"{song}_Separation_{time.strftime('%Y%m%d-%H%M%S')}.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fn, b in files.items():
            z.writestr(fn, b)
        z.writestr(f"{song}_Separation.json", json.dumps({**report, "separation": proj["separation"]}, indent=2, default=str))
    (ex / name).write_bytes(buf.getvalue())
    return {"file": name, "report": report}


# ------------------------------------------------------------------ export
def export_project(job, pid: str, opts: dict):
    proj = load_project(pid)
    d = pdir(pid) / "renders" / "full"
    if not (d / "render.json").exists():
        raise ValueError("Render the full vocal first.")
    meta = json.loads((d / "render.json").read_text())
    eng = meta["engine"]
    if proj.get("commercial_mode"):
        ok, why = commercial_allowed(eng["license_id"])
        if not ok:
            raise EngineUnavailable(why)
    job.update(0.1, "Loading stems")
    stems = {}
    sr = meta["sample_rate"]
    for stem in meta.get("audio_sha256", {}):
        y, sr = sf.read(str(d / f"{stem}.wav"), dtype="float32")
        stems[stem] = y
    sep = pdir(pid) / "separation" / "instrumental.wav"
    if opts.get("include_instrumental") and sep.exists():
        from .audio_io import resample
        y, isr = sf.read(str(sep), dtype="float32", always_2d=True)
        stems["instrumental"] = resample(y, isr, sr)
    perf, _ = flavour_preview(proj)
    session = {
        "app": "Kuralkoodu", "app_version": __version__, "project": proj["name"], "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(), "engine": eng, "model_license": REGISTRY.get(eng["license_id"]),
        "seed": meta["seed"], "voice": meta["voice"], "flavour": meta["flavour"], "changes": meta["changes"],
        "tempo": perf["tempo"], "tempo_map": perf.get("tempo_map"), "key": perf["key"], "transpose": proj.get("transpose", 0),
        "lyrics": proj["lyrics"], "input": proj["input"], "input_type": proj["input_type"],
        "render_audio_sha256": meta["audio_sha256"], "commercial_mode": proj.get("commercial_mode", False),
    }
    job.update(0.4, "Writing WAV / MIDI / session")
    data, report = build_export(stems, sr, {**opts, "song_name": proj.get("song_name") or proj["name"]}, session,
                                perf["notes"], perf.get("tempo_map") or [{"time": 0, "bpm": perf["tempo"]["bpm"]}], perf["key"])
    ex = pdir(pid) / "exports"
    ex.mkdir(exist_ok=True)
    name = f"{proj['id']}_{time.strftime('%Y%m%d-%H%M%S')}.zip"
    (ex / name).write_bytes(data)
    return {"zip": name, "report": report}
