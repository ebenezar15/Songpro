# Kuralkoodu (குரல்கூடு)

*Kural* is Tamil for voice, *koodu* for nest: a home for voices.

Hum a melody (or sing one), type the lyrics in **Tamil** or English, and get a lead vocal
plus sample-aligned stems, MIDI and a session file ready for Logic Pro. Bring a finished song
and Kuralkoodu separates the vocal from the music, so you can export either part on its own.
Songs of 4 minutes and longer are supported end to end. Everything runs locally: no uploads,
no telemetry, and no model downloads unless you agree to them.

```bash
./run.sh            # creates .venv, installs pinned deps, builds the UI, serves http://127.0.0.1:8765
```

Development: `cd backend && python -m kuralkoodu.server`, and `cd frontend && npm run dev` (Vite proxies `/api`).
Tests: `cd backend && python -m pytest -q`. Measurements: `python backend/scripts/acceptance.py samples`.

## What works today

| Step | Status |
|---|---|
| Import WAV / AIFF / MP3 / M4A / mic recording (ffmpeg decode, never fails silently) | done |
| Input types: hum + typed lyrics, **full song with music** (separates first), sung vocal (keeps the words), voice sample (identity only) | done |
| **Vocal separation**: built-in (no download) or Demucs htdemucs (after consent); export the isolated vocal and/or instrumental as WAV (44.1/48/96 kHz, 24-bit or 32-bit float, original channels / mono / stereo); include the instrumental as a stem in the Logic export | done (Demucs path written, not testable here: its weight host is blocked in this container) |
| **Long songs**: analysis, synthesis, voice conversion and separation run in blocks; tested at 4 minutes | done |
| **Studio UI**: one screen with a shared timeline (song, isolated vocal, instrumental, notes and pitch, new lead, doubles, harmonies, ad-libs), mute/solo, sample-synced playback with Original / Vocal only / New vocal / New + music compare | done |
| Analysis: F0 (YIN), notes, tempo + tempo map, key, loudness, vibrato, slides, sustains, phrases, breaths, confidences, low-confidence flags, "contains music?" check | done |
| Editable piano roll (move, resize, add, delete, edit lyrics; F0 thread, beat grid, loudness lane, shaded low-confidence areas). Edits override detection at render time | done |
| Voice: Male/Female × Child-like…Senior timbre presets, built-in parametric voice bank, consented user reference, Natural Range check with octave / key suggestions | done |
| Flavour engine: Faithful↔Creative master, 9 controls, 7 presets, Original / Natural / Hard tuning, Carnatic raga scales, seed + lock, doubles / harmonies / ad-libs on their own stems, change list with per-phrase undo | done |
| Render an 8-bar preview (from the playhead) or the full song as cancelable background jobs with progress + logs | done |
| Logic export: 44.1/48/96 kHz, 24-bit PCM or 32-bit float, mono/stereo, dry + optional processed lead, stems, BWF timestamp, MIDI (notes, lyrics, tempo map), JSON session, Logic import guide, common-gain anti-clip (no normalise/limit/truncate) | done |
| Licence registry + Commercial Mode export block | done |
| Projects saved to `~/Kuralkoodu/projects` (analysis cached by audio hash) | done |
| Neural engines (SoulX-Singer, Seed-VC), RMVPE/FCPE, Demucs | adapters + consent flow; **needs wiring and testing on your Mac** (Milestone 4) |
| Tauri desktop packaging | not started (Milestone 7) |

## Tamil: how "clean regional pronunciation" is handled

`backend/kuralkoodu/lyrics/tamil.py` turns Tamil script (or Tanglish) into IPA syllables using
the context rules Tamil speakers apply without thinking, which English-trained singers get wrong:

* **One letter, three sounds.** க ச ட த ப are voiceless and unaspirated at the start of a word or when doubled
  (பக்கம் *pakkam*), voiced after a nasal (அன்பே *anbē*, எங்கே *eŋgē*), and softened between vowels
  (அது *aðɨ*, பாடல் *pāɖal*, காதல் *kāðal*).
* **Retroflexes.** ழ = ɻ, ள = ɭ, ண = ɳ, ட = ʈ/ɖ. The synthesiser lowers F3 for these, which is the
  acoustic cue listeners use.
* **ர** is a tap ɾ and **ற** a trill r. ற்ற → *tr* (வெற்றி *vetri*) and ன்ற → *ndr* (ஒன்று *ondrɨ*).
* **ந** is dental n̪ and **ன** is alveolar n. **த** is dental t̪.
* **Word-final short உ** becomes the unrounded ɨ (அது, பஞ்சு).
* Options: word-initial **ச** as *s* (spoken / film standard) or *tʃ* (literary / Sri Lankan); colloquial
  **எ → ye** glide.
* The Import screen shows every syllable with its IPA, so you can check pronunciation before rendering.
* Tamil-relevant extras: a **Gamaka** control (kampita oscillation toward the upper scale note), an
  *Indian Classical-inspired* preset, raga scales (Shankarabharanam, Kalyani, Kharaharapriya, Mayamalavagowla,
  Mohanam, Hindolam and others) for hard tune, harmonies and ornaments, and Tamil ad-lib vocables (ஆ, ம், ஓ, லா).

For a release-quality Tamil vocal, the most authentic route is **Sung vocal** input: sing or speak the Tamil
line yourself, then change only the voice (the local *Voice Character* DSP engine now; Seed-VC later). The
pronunciation then comes from a real Tamil speaker.

## Engines (`VocalEngine` interface, `backend/kuralkoodu/engines/`)

| Engine | Purpose | Licence | Status |
|---|---|---|---|
| `formant` Kuralkoodu Formant Singer | Lyrics + melody → vocal, all Tamil phonemes, deterministic | this repo | works on CPU, 0.18× real time. **Sounds clearly synthetic.** Use it as a pronunciation-accurate guide |
| `dsp-convert` Voice Character | Sung vocal → new timbre (formant + transpose + breath), keeps words | this repo | works; extreme gender flips sound processed |
| `mock` | Instant test tones | this repo | works |
| `soulx-singer` | Neural score/F0 + lyrics → vocal, zero-shot timbre | Apache-2.0 code and weights | adapter + consent; Tamil only as approximate cross-lingual phonemes (trained on Mandarin, English, Cantonese) |
| `seed-vc` | Singing voice conversion | **unclear** (GPL-3.0 reported by mirrors) | adapter + consent; blocked in Commercial Mode |

Model files stay out of git (`.gitignore`). Each one is downloaded only after the UI shows its licence and you click consent.

## Vocal separation (`backend/kuralkoodu/separation.py`)

**Quick (built-in)** needs no download and is deterministic:

1. Track the lead melody (harmonic-summation pitch salience), then keep only the singer's harmonics with a comb mask.
2. REPET-SIM: music repeats, so each moment's background is the median of the most similar moments elsewhere in the song
   (searched within ±45 s, so long songs stay fast). Non-repeating energy above 3.5 kHz is kept as consonants and breaths.
3. Gently favour centre-panned, in-phase content, and roll off the bass and kick region.
4. The instrumental is *mix − vocal*, so the two parts always add back up to the original exactly.

**Studio (Demucs htdemucs)**, MIT licence, about 80 MB of weights, PyTorch with MPS on Apple Silicon. It is much cleaner on
real, dense mixes. It runs only after you install `requirements-ml.txt` and click consent.

When a full song is analysed, the isolated vocal is used, with stricter gates: quieter bleed and notes far outside the
singer's register are ignored, and the app tells you how many it dropped.

## Research notes (checked 2026-11-03)

* SoulX-Singer: Apache-2.0 for code and the official weights; Mandarin, English and Cantonese training data; weights on
  Hugging Face; some third-party mirrors bundle a GPL-3.0 dereverb model and an unlicensed karaoke model, so use the
  official release. [GitHub](https://github.com/Soul-AILab/SoulX-Singer) ·
  [a Hugging Face copy of the model card](https://huggingface.co/AlexthePromptCoder/SoulX-Singer) ·
  [mirror notes](https://huggingface.co/spoonn/SoulX-Singer-Native-ComfyUI)
* Seed-VC: zero-shot voice and singing conversion (F0-conditioned 44.1 kHz model). The licence is unclear: mirrors say
  GPL-3.0, but I could not confirm it from the upstream LICENSE. [Repo](https://github.com/Plachtaa/seed-vc) ·
  [paper](https://ar5iv.arxiv.org/html/2411.09943) · [Gitee mirror](https://gitee.com/LeeBaye/seed-vc)
* FCPE / torchfcpe: MIT for code and bundled weights. [PyPI](https://pypi.org/project/torchfcpe/0.0.3) ·
  [repo](https://uithub.com/CNChTu/FCPE). RMVPE: licence not confirmed. [Repo](https://github.com/Dream-High/RMVPE)
* Demucs: MIT (from the upstream repository, not re-verified in this session's search).
* There is **no open-source Tamil singing model**. Open Tamil *speech* data exists:
  [OpenSLR-65](https://www.openslr.org/65) (CC BY-SA 4.0, about 7 h). A DiffSinger-style model fine-tuned on
  consented Tamil singing would be the long-term path.

## Measured results (`samples/acceptance_report.json`, Linux CPU container, not your Mac)

| Check | Target | Result |
|---|---|---|
| Median pitch error, clean generated hums (6 random melodies, 72–150 BPM, 0–40 cent vibrato) | < 25 cents | **2.2 cents** (p95 5.2) |
| Tempo error | < 1 BPM | **max 0.15 BPM** |
| Unseen 31 s hum + Tamil lyrics → vocal | usable vocal | rendered in 5.5 s (0.18× real time), peak RSS 518 MB, 104.05 BPM detected (true 104) |
| Creative = 0: re-analyse the render and compare with the performance | close | 42 of 45 notes recovered; median pitch error 1.5 cents (p95 4.9); median onset error 10 ms (p95 30 ms) |
| Same input + settings + seed + engine version | identical | **bit-identical** for formant / DSP / mock engines. Neural engines are flagged `deterministic: false` |
| Stems sample-aligned, valid headers | aligned | all stems have identical frame counts, at 48 kHz/24-bit, with BWF `bext` first |
| **4-minute song** (242 s hum, 490 notes) | works | analysis 3.4 s (122.0 BPM, true 122); full render 14.7 s; 242.2 s of output. Peak memory 674 MB in a render-only run |
| **Separation, synthetic songs** (known vocal over a repeating stereo band, 3 songs) | better than the mix | vocal SDR −0.5 → **+17 to +19 dB**; instrumental +0.5 → **+17 to +18 dB**; vocal + instrumental equals the mix exactly |
| **Separation, 4 min 5 s stereo** | works | 25 s on 4 CPU cores, peak about 1.2 GB |
| Melody from a separated vocal (3 synthetic songs) | close | 35 of 40, 39 of 48 and 43 of 50 true notes recovered; a few extra notes remain |

The 3 "missing" notes at Creative = 0 are short repeated notes that the re-analysis merges. The render keeps them.
The separation numbers come from synthetic songs, which are much easier than real mixes with reverb, doubled vocals
and dense arrangements. Expect less on real music, especially with the quick method.
Automated tests (`pytest`, 33 passing) cover analysis accuracy, Tamil allophony, alignment and melisma, flavour
determinism and per-phrase undo isolation, Creative-0 fidelity, WAV/BWF/MIDI headers, the export gain rule, separation
quality and exact complement, the full-song API flow (separate → export vocals alone → analyse → render → Logic export with
the instrumental), and a 4-minute song rendered through the API.

**None of this says the vocal sounds good.** Please listen to `samples/*.flac` (lossless copies of the WAV renders)
and tell me what you hear:
`unseen_tamil_lead_creative0`, `unseen_tamil_lead_classical` (gamaka + slides + melisma),
`unseen_tamil_lead_male_adult`, and the input `unseen_hum_30s`. For separation:
`separation_input_song_30s`, `separation_isolated_vocal`, `separation_instrumental`.

## Known limitations

* The local formant engine is intelligible but robotic. Consonants are weak, and it has no real vocal identity. It exists so
  the whole workflow, including Tamil pronunciation rules, can be tested and used without downloads.
* The analysis uses YIN. It is accurate on clean hums. On full songs it works from the separated vocal, which still has
  some bleed with the quick method, so check the notes lane (unsure areas are shaded red) or use Demucs.
* The quick separator follows one lead melody. Backing vocals, wide stereo vocal effects and long reverb tails go mostly
  to the instrumental, and loud sustained instruments in the vocal range can leak into the vocal.
* English G2P is rule-based (no pronouncing dictionary), so English is weaker than Tamil in the local engine.
* SoulX-Singer and Seed-VC adapters assume an upstream CLI. They must be pinned to a commit and verified on the Mac
  (Milestone 4). Tamil through SoulX-Singer will lose some retroflex and dental contrasts.
* This was built and measured in a Linux cloud container, so the MPS / Apple-Silicon checks in the brief still need to
  run on your machine (`GET /api/system` reports chip, RAM, torch, MPS).
* Range check uses fixed comfortable ranges per preset, not a measured singer.

## Layout

```
backend/kuralkoodu/
  analysis.py      F0, notes, tempo, key, dynamics, expression, confidence
  lyrics/          Tamil (script + Tanglish) and English G2P, syllable→note alignment
  performance layer: service.py (project + edits), flavour.py (seeded expression), planner.py (frame-level plan)
  phonemes.py      acoustic targets per phoneme   voices.py  age/gender presets, range check, reference timbre
  engines/         base, formant, dsp_convert, mock, neural (SoulX-Singer, Seed-VC)
  export.py        WAV/BWF writer, MIDI, processing, zip   licenses.py  registry + Commercial Mode
  jobs.py, server.py (FastAPI, 127.0.0.1)
  separation.py    built-in separator + Demucs adapter
frontend/src/      React + TypeScript studio: Timeline (ruler + waveform lanes), PianoRoll (canvas),
                   Panels (source, separate, voice, flavour, export), useMixer (shared Web Audio clock)
samples/           renders + acceptance report
```

## Next milestones (waiting for your approval)

1. Run `/api/system` and the spike on your Mac (chip, RAM, MPS), then pin and wire SoulX-Singer and measure it.
2. Decide on Seed-VC after confirming its upstream licence.
3. Tauri packaging with the Python backend as a sidecar.
