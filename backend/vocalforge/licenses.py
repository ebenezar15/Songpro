"""Licence registry for engines, models and helper tools. Verified 2026-11-03 from the
upstream pages listed in `source`. 'unclear' blocks export in Commercial Mode."""
from __future__ import annotations

REGISTRY = {
    "vocalforge": {
        "name": "VocalForge built-in engines and voice bank", "code": "MIT (this repository)", "weights": "none (parametric)",
        "commercial": "allowed", "source": "this repository", "size_mb": 0, "notes": "No recorded voice data is used."},
    "soulx-singer": {
        "name": "SoulX-Singer (Soul-AILab)", "code": "Apache-2.0", "weights": "Apache-2.0 (official Hugging Face release)",
        "commercial": "allowed", "source": "https://github.com/Soul-AILab/SoulX-Singer ; https://huggingface.co/Soul-AILab/SoulX-Singer",
        "size_mb": None, "languages": ["zh", "en", "yue"],
        "notes": "Trained on Mandarin, English and Cantonese. Tamil must be sent as cross-lingual phonemes: retroflex and "
                 "dental contrasts may be weakened. Third-party mirrors can bundle GPL-3.0 extras (e.g. dereverb); "
                 "use the official release. README states intended use: research, education, legitimate applications."},
    "seed-vc": {
        "name": "Seed-VC (Plachtaa)", "code": "unclear (GPL-3.0 reported by mirrors; confirm upstream LICENSE)",
        "weights": "unclear", "commercial": "unclear", "source": "https://github.com/Plachtaa/seed-vc",
        "size_mb": None,
        "notes": "Zero-shot voice / singing voice conversion (F0-conditioned 44.1 kHz model). Best route for authentic Tamil: "
                 "sing or speak the Tamil guide yourself, convert only the timbre. Blocked in Commercial Mode until verified."},
    "rmvpe": {
        "name": "RMVPE pitch estimator", "code": "unclear (verify upstream)", "weights": "unclear (community mirrors)",
        "commercial": "unclear", "source": "https://github.com/Dream-High/RMVPE", "size_mb": 180,
        "notes": "Optional polyphonic-robust F0. The built-in YIN tracker is used when absent."},
    "fcpe": {
        "name": "FCPE / torchfcpe", "code": "MIT", "weights": "MIT (bundled with torchfcpe 0.0.4)", "commercial": "allowed",
        "source": "https://github.com/CNChTu/FCPE", "size_mb": 40, "notes": "Optional fast neural F0."},
    "demucs": {
        "name": "Demucs (Meta)", "code": "MIT", "weights": "MIT (htdemucs)", "commercial": "allowed",
        "source": "https://github.com/facebookresearch/demucs", "size_mb": 320,
        "notes": "Vocal / accompaniment separation. The repository is archived; the code still works with pinned torch."},
}


def commercial_allowed(license_id: str) -> tuple[bool, str]:
    rec = REGISTRY.get(license_id)
    if rec is None:
        return False, f"'{license_id}' is not in the licence registry."
    if rec["commercial"] != "allowed":
        return False, f"{rec['name']}: commercial use is '{rec['commercial']}'. Export is blocked in Commercial Mode."
    return True, "ok"
