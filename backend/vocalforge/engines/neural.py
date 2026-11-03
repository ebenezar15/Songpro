"""Adapters for neural engines. They are opt-in: model files are never in git and are
downloaded only after the user has seen size + licence and given consent.

The exact inference entry points of SoulX-Singer / Seed-VC change between releases, so
these adapters prepare engine-ready inputs (MIDI notes, F0 curve, phoneme lyrics, reference
voice) and call the upstream inference script that the user installed. Wiring to a pinned
upstream commit is Milestone 4 and must be verified on the target Mac first.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from ..licenses import REGISTRY
from .base import EngineUnavailable, VocalEngine

# Tamil -> phones the English/Mandarin-trained SoulX-Singer knows (approximate, lossy)
TAMIL_TO_CROSSLINGUAL = {
    "ʈ": "t", "ɖ": "d", "ɳ": "n", "ɭ": "l", "ɻ": "r", "t̪": "t", "d̪": "d", "n̪": "n", "ð": "dh", "β": "v",
    "ɣ": "g", "ɾ": "r", "r": "r", "ʋ": "v", "ɲ": "ny", "ŋ": "ng", "ʂ": "sh", "ɨ": "ih",
    "aː": "aa", "iː": "iy", "uː": "uw", "eː": "ey", "oː": "ow", "a": "ah", "i": "ih", "u": "uh", "e": "eh", "o": "ao",
}


def models_dir() -> Path:
    return Path(os.environ.get("VOCALFORGE_HOME", Path.home() / "VocalForge")) / "models"


def consent_given(model_id: str) -> bool:
    f = models_dir().parent / "consents.json"
    if not f.exists():
        return False
    return bool(json.loads(f.read_text()).get(model_id))


def record_consent(model_id: str, value: bool = True):
    f = models_dir().parent / "consents.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(f.read_text()) if f.exists() else {}
    data[model_id] = value
    f.write_text(json.dumps(data, indent=2))


class _NeuralBase(VocalEngine):
    repo_env = ""
    hf_repo = ""

    def model_path(self) -> Path:
        return models_dir() / self.id

    def available(self):
        rec = REGISTRY[self.license_id]
        if importlib.util.find_spec("torch") is None:
            return False, "PyTorch is not installed (pip install -r requirements-ml.txt)."
        if not consent_given(self.license_id):
            return False, f"Needs your consent to download ({rec['weights']}; see licence registry)."
        if not self.model_path().exists():
            return False, f"Model files not found in {self.model_path()}."
        if not os.environ.get(self.repo_env):
            return False, f"Set {self.repo_env} to your local checkout of {rec['source'].split(' ;')[0]}."
        return True, "ready"

    def download(self):
        if not consent_given(self.license_id):
            raise EngineUnavailable("Consent required before download.")
        if importlib.util.find_spec("huggingface_hub") is None:
            raise EngineUnavailable("pip install huggingface_hub to download models.")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=self.hf_repo, local_dir=str(self.model_path()))


class SoulXSingerEngine(_NeuralBase):
    id = "soulx-singer"
    name = "SoulX-Singer (neural SVS)"
    license_id = "soulx-singer"
    languages = ("en", "zh", "yue", "ta*")
    native_sr = 44100
    repo_env = "SOULX_SINGER_REPO"
    hf_repo = "Soul-AILab/SoulX-Singer"
    deterministic = False  # diffusion/flow sampling; seeded, but GPU kernels may vary
    version = "upstream"

    def prepare_inputs(self, plan, workdir: Path, reference_wav: str | None):
        workdir.mkdir(parents=True, exist_ok=True)
        notes = []
        for n in plan["notes"]:
            ph = n.get("phonemes") or {}
            seq = ph.get("onset", []) + [ph.get("vowel", "a")] + ph.get("coda", [])
            notes.append({"start": n["start"], "dur": n["dur"], "midi": n["midi"],
                          "phones": [TAMIL_TO_CROSSLINGUAL.get(p, p) for p in seq], "lyric": n.get("lyric", "")})
        (workdir / "score.json").write_text(json.dumps({"notes": notes}, ensure_ascii=False, indent=1))
        np.save(workdir / "f0.npy", np.asarray(plan["f0_hz"], np.float32))
        if reference_wav:
            shutil.copy(reference_wav, workdir / "reference.wav")
        return workdir

    def render(self, plan, seed, progress=lambda f, m: None, cancel=lambda: False):
        ok, msg = self.available()
        if not ok:
            raise EngineUnavailable(msg)
        work = self.prepare_inputs(plan, models_dir().parent / "tmp" / "soulx", plan.get("reference_wav"))
        script = Path(os.environ[self.repo_env]) / "infer.py"
        if not script.exists():
            raise EngineUnavailable(f"{script} not found; point {self.repo_env} at the SoulX-Singer checkout.")
        out = work / "out.wav"
        proc = subprocess.run(["python", str(script), "--score", str(work / "score.json"), "--f0", str(work / "f0.npy"),
                               "--ref", str(work / "reference.wav"), "--seed", str(seed), "--out", str(out),
                               "--ckpt", str(self.model_path())], capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            raise EngineUnavailable("SoulX-Singer inference failed: " + proc.stderr[-800:])
        y, sr = sf.read(str(out), dtype="float32", always_2d=True)
        return y.mean(axis=1), sr


class SeedVCEngine(_NeuralBase):
    id = "seed-vc"
    name = "Seed-VC (singing voice conversion)"
    license_id = "seed-vc"
    languages = ("any (keeps the sung words)",)
    input_types = ("sung",)
    native_sr = 44100
    repo_env = "SEED_VC_REPO"
    hf_repo = "Plachta/Seed-VC"
    deterministic = False
    version = "upstream"

    def render(self, plan, seed, progress=lambda f, m: None, cancel=lambda: False):
        ok, msg = self.available()
        if not ok:
            raise EngineUnavailable(msg)
        src, ref = plan.get("source_wav"), plan.get("reference_wav")
        if not src or not ref:
            raise EngineUnavailable("Seed-VC needs a sung source vocal and a consented reference voice.")
        out_dir = models_dir().parent / "tmp" / "seedvc"
        out_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(["python", str(Path(os.environ[self.repo_env]) / "inference.py"), "--source", src,
                               "--target", ref, "--output", str(out_dir), "--f0-condition", "True",
                               "--semi-tone-shift", str(int(plan.get("transpose", 0)))], capture_output=True, text=True)
        wavs = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        if proc.returncode != 0 or not wavs:
            raise EngineUnavailable("Seed-VC inference failed: " + proc.stderr[-800:])
        y, sr = sf.read(str(wavs[-1]), dtype="float32", always_2d=True)
        return y.mean(axis=1), sr
