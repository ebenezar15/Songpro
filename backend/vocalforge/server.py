"""FastAPI app. Binds to 127.0.0.1 only; no telemetry, no uploads anywhere."""
from __future__ import annotations

import copy
import os
import platform
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__, service
from .audio_io import AudioImportError
from .engines import ENGINES
from .engines.neural import consent_given, record_consent
from .export import LOGIC_GUIDE
from .flavour import CONTROLS, PRESETS, SCALES
from .jobs import JobManager
from .licenses import REGISTRY
from .lyrics import LANGUAGES, parse_lyrics
from .separation import separation_status
from .voices import AGE_LABELS, AGES, GENDERS, LICENSE_NOTE, VOICE_BANK, analyse_reference, range_check

app = FastAPI(title="VocalForge", version=__version__)
jobs = JobManager()


def _proj(pid):
    try:
        return service.load_project(pid)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "Project not found")


@app.get("/api/health")
def health():
    return {"ok": True, "version": __version__}


@app.get("/api/system")
def system():
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "cpu_count": os.cpu_count()}
    try:
        import torch  # noqa
        info["torch"] = torch.__version__
        info["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        info["cuda"] = torch.cuda.is_available()
    except Exception:
        info["torch"] = None
        info["mps"] = False
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except Exception:
        pass
    return info


@app.get("/api/meta")
def meta():
    return {"engines": [e.info() for e in ENGINES.values()], "licenses": REGISTRY, "presets": PRESETS, "controls": CONTROLS,
            "scales": list(SCALES.keys()), "languages": LANGUAGES, "ages": [{"id": a, "label": AGE_LABELS[a]} for a in AGES],
            "genders": GENDERS, "voice_bank": VOICE_BANK, "voice_bank_note": LICENSE_NOTE,
            "separation": dict(zip(("available", "status"), separation_status())), "logic_guide": LOGIC_GUIDE}


class ConsentIn(BaseModel):
    consent: bool


@app.post("/api/models/{model_id}/consent")
def model_consent(model_id: str, body: ConsentIn):
    if model_id not in REGISTRY:
        raise HTTPException(404, "unknown model")
    record_consent(model_id, body.consent)
    return {"model": model_id, "consent": consent_given(model_id), "license": REGISTRY[model_id]}


# ------------------------------------------------------------- projects
class NewProject(BaseModel):
    name: str = "Untitled"


@app.get("/api/projects")
def projects():
    return service.list_projects()


@app.post("/api/projects")
def new_project(body: NewProject):
    return service.create_project(body.name)


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    return _proj(pid)


ALLOWED_UPDATE = {"name", "song_name", "input_type", "lyrics", "voice", "flavour", "commercial_mode", "transpose"}


@app.put("/api/projects/{pid}")
def update_project(pid: str, body: dict):
    proj = _proj(pid)
    for k, v in body.items():
        if k in ALLOWED_UPDATE:
            proj[k] = v
    if "reference_use" in body and proj.get("reference"):
        proj["reference"]["use"] = bool(body["reference_use"])
    if "lyrics" in body and proj.get("performance"):
        service.apply_lyrics(proj)
    service.save_project(proj)
    return proj


@app.put("/api/projects/{pid}/performance")
def update_performance(pid: str, body: dict):
    """User edits to notes / tempo / key override detected values at render time."""
    proj = _proj(pid)
    if not proj.get("performance"):
        raise HTTPException(400, "Analyse first")
    perf = proj["performance"]
    for k in ("notes", "tempo", "key", "tempo_map"):
        if k in body:
            perf[k] = body[k]
    perf["edited"] = True
    service.renumber_phrases(perf)
    if body.get("realign_lyrics"):
        service.apply_lyrics(proj)
    service.save_project(proj)
    return proj


@app.post("/api/projects/{pid}/performance/reset")
def reset_performance(pid: str):
    proj = _proj(pid)
    a = service.load_analysis(pid)
    if not a:
        raise HTTPException(400, "No analysis")
    proj["performance"] = service.performance_from_analysis(a)
    service.apply_lyrics(proj)
    service.save_project(proj)
    return proj


@app.post("/api/projects/{pid}/audio")
async def upload_audio(pid: str, file: UploadFile = File(...), role: str = Form("input"), consent: bool = Form(False)):
    proj = _proj(pid)
    if role not in ("input", "reference"):
        raise HTTPException(400, "role must be input or reference")
    if role == "reference" and not consent:
        raise HTTPException(400, "Please confirm you own this voice or have permission to use it.")
    data = await file.read()
    try:
        info = service.store_audio(pid, file.filename or "upload.wav", data, role)
    except AudioImportError as e:
        raise HTTPException(400, str(e))
    if role == "input":
        proj["input"] = info
        proj["performance"] = None
    else:
        from .audio_io import load_audio
        x, sr = load_audio(service.pdir(pid) / "reference_decoded.wav")
        try:
            info["timbre"] = analyse_reference(x, sr)
        except ValueError as e:
            raise HTTPException(400, str(e))
        info.update(consent=True, use=True)
        proj["reference"] = info
    service.save_project(proj)
    return proj


@app.post("/api/projects/{pid}/analyze")
def analyze(pid: str, force: bool = False, use_separated: bool = False):
    _proj(pid)
    return jobs.submit("analysis", pid, service.run_analysis, pid, force, use_separated).to_dict()


@app.get("/api/projects/{pid}/analysis")
def get_analysis(pid: str):
    a = service.load_analysis(pid)
    if not a:
        raise HTTPException(404, "Not analysed yet")
    return a


@app.post("/api/projects/{pid}/separate")
def separate(pid: str):
    _proj(pid)

    def run(job, pid):
        from .audio_io import load_audio, to_mono, write_preview_wav
        from .separation import separate_vocals
        job.update(0.1, "Running Demucs (local)")
        out = separate_vocals(service.pdir(pid) / "input_decoded.wav", service.pdir(pid) / "separation")
        x, sr = load_audio(out)
        write_preview_wav(service.pdir(pid) / "separated_decoded.wav", to_mono(x), sr)
        return {"file": "separated_decoded.wav"}

    return jobs.submit("separation", pid, run, pid).to_dict()


class LyricsIn(BaseModel):
    text: str
    language: str = "ta"
    dialect: str = "standard"
    glides: bool = True


@app.post("/api/lyrics/parse")
def lyrics_parse(body: LyricsIn):
    return {"syllables": parse_lyrics(body.text, body.language, body.dialect, body.glides)}


@app.get("/api/projects/{pid}/flavour")
def flavour_preview(pid: str):
    proj = _proj(pid)
    if not proj.get("performance"):
        raise HTTPException(400, "Analyse first")
    perf, changes = service.flavour_preview(proj)
    return {"changes": changes, "notes": perf["notes"], "range": range_check(perf["notes"], service.current_voice(proj)),
            "voice": service.current_voice(proj)}


class RenderIn(BaseModel):
    mode: str = "preview"
    start_bar: int = 0
    bars: int = 8


@app.post("/api/projects/{pid}/render")
def render(pid: str, body: RenderIn):
    _proj(pid)
    if body.mode not in ("preview", "full"):
        raise HTTPException(400, "mode must be preview or full")
    return jobs.submit(f"render-{body.mode}", pid, service.render_project, pid, body.mode, body.start_bar, body.bars).to_dict()


@app.get("/api/projects/{pid}/renders/{mode}")
def render_meta(pid: str, mode: str):
    f = service.pdir(pid) / "renders" / mode / "render.json"
    if mode not in ("preview", "full") or not f.exists():
        raise HTTPException(404, "No render")
    return JSONResponse(content=__import__("json").loads(f.read_text()))


@app.get("/api/projects/{pid}/renders/{mode}/{stem}.wav")
def render_audio(pid: str, mode: str, stem: str):
    if mode not in ("preview", "full") or not stem.replace("_", "").isalnum():
        raise HTTPException(400, "bad path")
    f = service.pdir(pid) / "renders" / mode / f"{stem}.wav"
    if not f.exists():
        raise HTTPException(404, "No such stem")
    return FileResponse(f, media_type="audio/wav")


@app.get("/api/projects/{pid}/input.wav")
def input_audio(pid: str, which: str = "input"):
    name = {"input": "input_decoded.wav", "reference": "reference_decoded.wav", "separated": "separated_decoded.wav"}.get(which)
    f = service.pdir(pid) / (name or "")
    if not name or not f.exists():
        raise HTTPException(404, "No audio")
    return FileResponse(f, media_type="audio/wav")


class ExportIn(BaseModel):
    sample_rate: int = 48000
    bit_depth: str = "24"
    channels: int = 1
    processed: bool = False
    bwf: bool = False
    bwf_time_reference_s: float = 0.0
    midi: bool = True
    stems: list[str] | None = None


@app.post("/api/projects/{pid}/export")
def export(pid: str, body: ExportIn):
    _proj(pid)
    if body.sample_rate not in (44100, 48000, 96000) or body.bit_depth not in ("24", "32f") or body.channels not in (1, 2):
        raise HTTPException(400, "Unsupported export format")
    return jobs.submit("export", pid, service.export_project, pid, body.model_dump()).to_dict()


@app.get("/api/projects/{pid}/exports/{name}")
def download_export(pid: str, name: str):
    if "/" in name or ".." in name or not name.endswith(".zip"):
        raise HTTPException(400, "bad name")
    f = service.pdir(pid) / "exports" / name
    if not f.exists():
        raise HTTPException(404, "No export")
    return FileResponse(f, media_type="application/zip", filename=name)


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "No job")
    return j.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "No job")
    j.cancel()
    return j.to_dict()


_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="ui")


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("VOCALFORGE_PORT", "8765")))


if __name__ == "__main__":
    main()
