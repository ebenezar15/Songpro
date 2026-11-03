"""Objective acceptance measurements (run: python scripts/acceptance.py [out_dir]).

1. Unseen 30-second hum (random melody/tempo/seed not used in unit tests) + typed Tamil lyrics -> vocal.
2. Pitch / tempo accuracy of analysis on clean generated files.
3. Creative = 0 fidelity: re-analyse the rendered vocal and compare with the edited performance.
4. Reproducibility: same input + settings + seed + engine version -> identical audio hash.
5. Export: stems sample-aligned, headers valid.
Prints a JSON report. It does NOT judge vocal quality: listen to the renders.
"""
import hashlib
import io
import json
import resource
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from synth import hum  # noqa: E402
from vocalforge.analysis import analyze  # noqa: E402
from vocalforge.engines import FormantSingerEngine  # noqa: E402
from vocalforge.export import build_export, read_wav_header  # noqa: E402
from vocalforge.flavour import apply_flavours, make_double, make_harmony  # noqa: E402
from vocalforge.lyrics import align_to_notes, parse_lyrics  # noqa: E402
from vocalforge.planner import build_plan  # noqa: E402
from vocalforge.service import performance_from_analysis, renumber_phrases  # noqa: E402
from vocalforge.voices import voice_params  # noqa: E402

LYRICS = """கண்ணே கலைமானே கன்னி மயிலென
கண்டேன் உன்னை நானே
அந்தி பகல் உன்னை நான் பார்க்கிறேன்
ஆண்டவன் எனக்கே தந்தான்
வெற்றி ஒன்று தமிழ் பாடல்"""


def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261103)
    report = {}

    # --- 2. accuracy sweep on clean generated files
    pitch_err, tempo_err = [], []
    for trial in range(6):
        bpm = float(rng.uniform(72, 150))
        mel = list(60 + rng.choice([0, 2, 4, 5, 7, 9, 11, 12], size=24))
        y, sr, _ = hum(mel, bpm=bpm, seed=trial, vibrato=float(rng.uniform(0, 40)))
        a = analyze(y[:, None], sr)
        tempo_err.append(abs(a.tempo["bpm"] - bpm))
        if len(a.notes) == len(mel):
            pitch_err += [abs(n["midi"] - m) * 100 for n, m in zip(a.notes, mel)]
        else:
            report.setdefault("note_count_mismatch", []).append({"trial": trial, "expected": len(mel), "got": len(a.notes)})
    report["analysis_accuracy"] = {"median_pitch_error_cents": round(float(np.median(pitch_err)), 2),
                                   "p95_pitch_error_cents": round(float(np.percentile(pitch_err, 95)), 2),
                                   "tempo_errors_bpm": [round(t, 3) for t in tempo_err],
                                   "max_tempo_error_bpm": round(float(max(tempo_err)), 3)}

    # --- 1. unseen 30 s hum
    bpm = 104.0
    steps = rng.choice([-2, -1, 0, 1, 2, 3, -3], size=52)
    mel = list(np.clip(64 + np.cumsum(steps) * 1, 57, 74))
    beats = rng.choice([0.5, 1.0, 1.0, 1.5, 2.0], size=52)
    y = []
    onsets = []
    t = 0.5
    sr = 44100
    for m, b in zip(mel, beats):
        seg, _, _ = hum([m], bpm=bpm, beats_per_note=float(b), lead_in=0.0, gap=0.05, seed=int(m), vibrato=20)
        seg = seg[: int(b * 60 / bpm * sr)]
        y.append(seg)
        onsets.append(t)
        t += len(seg) / sr
        if t > 30:
            break
    y = np.concatenate([np.zeros(int(0.5 * sr), np.float32)] + y + [np.zeros(int(0.5 * sr), np.float32)])
    sf.write(out_dir / "unseen_hum_30s.wav", y, sr, subtype="PCM_24")
    t0 = time.time()
    a = analyze(y[:, None], sr).to_dict()
    t_an = time.time() - t0
    perf = performance_from_analysis(a)
    perf["notes"], align_rep = align_to_notes(perf["notes"], parse_lyrics(LYRICS, "ta"))
    renumber_phrases(perf)
    voice = voice_params("female", "young_adult", "meera")
    eng = FormantSingerEngine()

    # --- 3. creative = 0 fidelity
    p0, ch0 = apply_flavours(perf, {"creative": 0, "tuning": "natural", "seed": 11})
    t0 = time.time()
    y0, osr = eng.render(build_plan(p0, voice, a, seed=11), 11)
    t_render = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    sf.write(out_dir / "unseen_tamil_lead_creative0.wav", y0 / max(1e-9, np.abs(y0).max()) * 0.7, osr, subtype="PCM_24")
    r = analyze(y0[:, None], osr)
    ref = sorted(p0["notes"], key=lambda n: n["start"])
    ec, et, matched = [], [], 0
    for n in ref:
        g = min(r.notes, key=lambda x: abs(x["start"] - n["start"]))
        if abs(g["start"] - n["start"]) < 0.15:
            matched += 1
            et.append(abs(g["start"] - n["start"]) * 1000)
            ec.append(abs(g["midi"] - round(n["midi"])) * 100)
    report["creative0_fidelity"] = {
        "performance_notes": len(ref), "detected_in_render": len(r.notes), "matched": matched,
        "median_pitch_error_cents": round(float(np.median(ec)), 2), "p95_pitch_error_cents": round(float(np.percentile(ec, 95)), 2),
        "median_onset_error_ms": round(float(np.median(et)), 1), "p95_onset_error_ms": round(float(np.percentile(et, 95)), 1),
        "flavour_changes": len(ch0)}

    # --- flavoured render + stems
    pf, chf = apply_flavours(perf, {"creative": 60, "preset": "Indian Classical-inspired", "vibrato": 20, "scoops": 20, "slides": 60,
                                     "melisma": 50, "breathiness": 10, "grit": 0, "timing": 10, "dynamics": 50, "gamaka": 80, "seed": 11})
    lead, _ = eng.render(build_plan(pf, voice, a, seed=11), 11)
    dbl, _ = eng.render(build_plan(make_double(pf, 11, 0), voice, seed=11), 11)
    har, _ = eng.render(build_plan(make_harmony(pf, 2), voice, seed=11), 11)
    sf.write(out_dir / "unseen_tamil_lead_classical.wav", lead / np.abs(lead).max() * 0.7, osr, subtype="PCM_24")
    male = voice_params("male", "adult", "kavi")
    pm = {**pf, "transpose": -12}
    ym, _ = eng.render(build_plan(pm, male, a, seed=11), 11)
    sf.write(out_dir / "unseen_tamil_lead_male_adult.wav", ym / np.abs(ym).max() * 0.7, osr, subtype="PCM_24")

    # --- 4. reproducibility
    y1, _ = eng.render(build_plan(pf, voice, a, seed=11), 11)
    report["reproducibility"] = {"engine": eng.version, "identical_hash": hashlib.sha256(y1.tobytes()).hexdigest() == hashlib.sha256(lead.tobytes()).hexdigest()}

    # --- 5. export alignment
    data, rep = build_export({"lead": lead, "doubles": dbl * 0.7, "harmonies": har * 0.6}, osr,
                             {"sample_rate": 48000, "bit_depth": "24", "processed": True, "bwf": True},
                             {"voice": voice, "tempo": a["tempo"], "seed": 11}, pf["notes"], a["tempo_map"], a["key"])
    z = zipfile.ZipFile(io.BytesIO(data))
    hdrs = {n: read_wav_header(z.read(n)) for n in z.namelist() if n.endswith(".wav")}
    report["export"] = {"files": sorted(z.namelist()), "frames": sorted({h["frames"] for h in hdrs.values()}),
                        "sample_rates": sorted({h["sample_rate"] for h in hdrs.values()}), "bits": sorted({h["bits"] for h in hdrs.values()}),
                        "gain_applied_db": rep["gain_applied_db"], "aligned": len({h["frames"] for h in hdrs.values()}) == 1}
    (out_dir / "export_example.zip").write_bytes(data)
    report["unseen_30s"] = {"input_seconds": round(len(y) / sr, 2), "detected_bpm": a["tempo"]["bpm"], "true_bpm": bpm,
                            "key": a["key"]["label"], "notes": len(a["notes"]), "lyrics_alignment": align_rep,
                            "analysis_runtime_s": round(t_an, 2), "render_runtime_s": round(t_render, 2),
                            "render_realtime_factor": round(t_render / (len(y0) / osr), 3), "peak_rss_mb": round(rss, 1),
                            "flavour_changes_classical": [c["summary"] for c in chf][:12]}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    (out_dir / "acceptance_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "samples")
