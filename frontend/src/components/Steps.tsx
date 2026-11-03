import { useEffect, useRef, useState } from "react";
import { api, Job, midiName, waitJob } from "../api";
import Transport from "./Transport";

type P = { proj: any; meta: any; save: (patch: any) => Promise<void>; reload: () => Promise<void>; runJob: (j: Promise<Job>) => Promise<Job> };

const TAMIL_SAMPLE = "நிலா நிலா ஓடி வா\nநில்லாமல் ஓடி வா\nமலை மேலே ஏறி வா\nமல்லிகைப் பூ கொண்டு வா";

// ------------------------------------------------------------------ Import
export function ImportStep({ proj, meta, save, reload, runJob }: P) {
  const [rec, setRec] = useState<MediaRecorder | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [lyrics, setLyrics] = useState(proj.lyrics);
  const [syll, setSyll] = useState<any[]>([]);
  const chunks = useRef<Blob[]>([]);

  useEffect(() => setLyrics(proj.lyrics), [proj.id]);
  useEffect(() => {
    const t = setTimeout(async () => {
      if (!lyrics.text.trim()) return setSyll([]);
      try { setSyll((await api.parseLyrics(lyrics.text, lyrics.language, lyrics.dialect, lyrics.glides)).syllables); } catch { /* shown on save */ }
    }, 250);
    return () => clearTimeout(t);
  }, [lyrics]);

  async function upload(file: Blob, name: string) {
    setBusy(true); setErr(null);
    try { await api.upload(proj.id, file, name, "input"); await reload(); }
    catch (e: any) { setErr(e.message); }
    finally { setBusy(false); }
  }
  async function toggleRec() {
    if (rec) { rec.stop(); setRec(null); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
      const r = new MediaRecorder(stream);
      chunks.current = [];
      r.ondataavailable = (e) => chunks.current.push(e.data);
      r.onstop = () => { stream.getTracks().forEach((t) => t.stop()); upload(new Blob(chunks.current, { type: r.mimeType }), "recording.webm"); };
      r.start();
      setRec(r);
    } catch (e: any) { setErr(`Microphone unavailable: ${e.message}`); }
  }
  async function analyse() {
    await save({ lyrics, input_type: proj.input_type });
    await runJob(api.analyze(proj.id));
    await reload();
  }
  const needsLyrics = proj.input_type === "hum";
  return (
    <div className="step-grid">
      <div>
        <h2>Bring in a melody</h2>
        <div className="drop" onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) upload(f, f.name); }}>
          <p>{proj.input ? <>Using <strong>{proj.input.original_name}</strong>, {proj.input.duration}s at {proj.input.sample_rate / 1000} kHz</> : "Drop a WAV, AIFF, MP3 or M4A file here"}</p>
          <div className="row">
            <label className="button">
              Choose file<input type="file" accept=".wav,.aif,.aiff,.mp3,.m4a,.flac" hidden
                onChange={(e) => { const f = e.target.files?.[0]; if (f) upload(f, f.name); }} />
            </label>
            <button className={rec ? "danger" : ""} onClick={toggleRec}>{rec ? "Stop recording" : "Record from microphone"}</button>
          </div>
          {busy && <p className="muted">Importing…</p>}
          {err && <p className="error" role="alert">{err}</p>}
        </div>
        <fieldset className="choice">
          <legend>What is this recording?</legend>
          {[
            ["hum", "Hummed or instrumental melody", "Uses the notes; you type the lyrics."],
            ["sung", "Sung vocal", "Keeps the existing words and performance."],
            ["voice_sample", "Voice sample", "Only used for voice identity, not melody."],
          ].map(([v, l, d]) => (
            <label key={v} className={`radio-card ${proj.input_type === v ? "on" : ""}`}>
              <input type="radio" name="itype" checked={proj.input_type === v} onChange={() => save({ input_type: v })} />
              <span><strong>{l}</strong><small>{d}</small></span>
            </label>
          ))}
        </fieldset>
      </div>
      <div>
        <h2>Lyrics</h2>
        <div className="row">
          <select value={lyrics.language} onChange={(e) => setLyrics({ ...lyrics, language: e.target.value })} aria-label="Language">
            {Object.entries(meta.languages).map(([k, v]) => <option key={k} value={k}>{v as string}</option>)}
          </select>
          {lyrics.language !== "en" && (
            <>
              <select value={lyrics.dialect} onChange={(e) => setLyrics({ ...lyrics, dialect: e.target.value })} aria-label="Pronunciation">
                <option value="standard">Spoken / film standard (ச → s)</option>
                <option value="literary">Literary / Sri Lankan (ச → tʃ)</option>
              </select>
              <label className="check"><input type="checkbox" checked={lyrics.glides} onChange={(e) => setLyrics({ ...lyrics, glides: e.target.checked })} /> Colloquial glides (எ → ye)</label>
            </>
          )}
        </div>
        <textarea className="lyrics" rows={6} value={lyrics.text} lang={lyrics.language === "ta" ? "ta" : undefined}
          placeholder={needsLyrics ? "Type the lyrics, one phrase per line. Use - to hold a syllable over extra notes." : "Optional for sung vocals"}
          onChange={(e) => setLyrics({ ...lyrics, text: e.target.value })} />
        {!lyrics.text && <button className="link" onClick={() => setLyrics({ ...lyrics, text: TAMIL_SAMPLE, language: "ta" })}>Try a Tamil example</button>}
        {syll.length > 0 && (
          <div className="syllables" aria-label="Pronunciation preview">
            {syll.map((s, i) => (
              <span key={i} className={`syl ${s.hold ? "hold" : ""} ${s.word_start ? "ws" : ""}`} title={s.word}>
                <b lang="ta">{s.text}</b><i>{s.hold ? "hold" : `/${s.ipa}/`}</i>
              </span>
            ))}
          </div>
        )}
        <p className="muted small">{syll.filter((s) => !s.hold).length} syllables. Each one is sung on one note, in order. Check the IPA: Tamil stops change sound by position (க is k / g / ɣ), and ழ ள ண are retroflex.</p>
        <button className="primary" disabled={!proj.input || (needsLyrics && !lyrics.text.trim())} onClick={analyse}>Analyse melody</button>
        {needsLyrics && !lyrics.text.trim() && proj.input && <p className="muted small">A hummed melody needs lyrics before analysis.</p>}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- Analysis
function Conf({ v, label }: { v: number; label: string }) {
  const lvl = v >= 0.6 ? "ok" : v >= 0.35 ? "mid" : "low";
  return (
    <div className={`conf ${lvl}`}>
      <span>{label}</span>
      <span className="bar" role="meter" aria-valuemin={0} aria-valuemax={1} aria-valuenow={v} aria-label={`${label} confidence`}><span style={{ width: `${Math.round(v * 100)}%` }} /></span>
      <b>{Math.round(v * 100)}%</b>
    </div>
  );
}

export function AnalysisStep({ proj, meta, analysis, reload, runJob, savePerf }: P & { analysis: any; savePerf: (b: any) => Promise<void> }) {
  const perf = proj.performance;
  const taps = useRef<number[]>([]);
  const [tap, setTap] = useState<number | null>(null);
  if (!analysis || !perf) return <p className="empty">Import audio and run the analysis to see tempo, key and notes.</p>;
  const keys: string[] = [];
  ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"].forEach((t, i) => { keys.push(`${t}|major|${i}`); keys.push(`${t}|minor|${i}`); });
  function tapTempo() {
    const now = performance.now();
    taps.current = [...taps.current.filter((t) => now - t < 3000), now];
    if (taps.current.length >= 3) {
      const d = taps.current.slice(1).map((t, i) => t - taps.current[i]);
      setTap(Math.round(60000 / (d.reduce((a, b) => a + b) / d.length) * 10) / 10);
    }
  }
  const k = perf.key;
  return (
    <div className="step-grid">
      <div>
        <h2>What we heard</h2>
        <div className="facts">
          <label>Tempo
            <span className="row">
              <input type="number" step={0.1} min={30} max={260} value={perf.tempo.bpm}
                onChange={(e) => savePerf({ tempo: { ...perf.tempo, bpm: +e.target.value } })} /> BPM
              <button onClick={tapTempo}>Tap</button>
              {tap && <button className="link" onClick={() => savePerf({ tempo: { ...perf.tempo, bpm: tap } })}>Use {tap}</button>}
            </span>
          </label>
          <label>Key
            <select value={`${k.tonic}|${k.mode}|${k.tonic_pc}`} onChange={(e) => {
              const [t, m, pc] = e.target.value.split("|");
              savePerf({ key: { ...k, tonic: t, mode: m, tonic_pc: +pc, label: `${t} ${m}`, short: `${t}${m === "major" ? "maj" : "min"}`, edited: true } });
            }}>
              {keys.map((v) => <option key={v} value={v}>{v.split("|").slice(0, 2).join(" ")}</option>)}
            </select>
          </label>
          <div>Notes <b>{perf.notes.length}</b>, phrases <b>{perf.phrases.length}</b>, breaths <b>{analysis.breaths.length}</b>, slides <b>{analysis.slides.length}</b></div>
          <div>Range <b>{midiName(Math.min(...perf.notes.map((n: any) => n.midi)))}–{midiName(Math.max(...perf.notes.map((n: any) => n.midi)))}</b></div>
        </div>
        <Conf v={analysis.tempo.confidence} label="Tempo" />
        <Conf v={analysis.key.confidence} label="Key" />
        <Conf v={analysis.note_confidence} label="Notes" />
        {analysis.tempo_map.length > 1 && <p className="small">Tempo changes at {analysis.tempo_map.slice(1).map((t: any) => `${t.time}s → ${t.bpm}`).join(", ")}</p>}
      </div>
      <div>
        <h2>Check these</h2>
        {analysis.warnings.length === 0 && analysis.low_confidence.length === 0 && <p className="muted">No problems found.</p>}
        {analysis.warnings.map((w: string, i: number) => <p key={i} className="warn">{w}</p>)}
        {analysis.music_likelihood > 0.5 && (
          <div className="callout">
            <p>This recording seems to include music. Separating the vocal first gives cleaner notes.</p>
            {meta.separation.available
              ? <button onClick={async () => { await runJob(api.separate(proj.id)); await runJob(api.analyze(proj.id, true, true)); await reload(); }}>Separate vocal and re-analyse</button>
              : <p className="small">{meta.separation.status} <button className="link" onClick={async () => { await api.consent("demucs", true); await reload(); }}>Allow Demucs download</button></p>}
          </div>
        )}
        {analysis.low_confidence.length > 0 && (
          <ul className="flags">
            {analysis.low_confidence.slice(0, 12).map((r: any, i: number) => <li key={i}><b>{r.start.toFixed(2)}–{r.end.toFixed(2)}s</b> {r.reason}</li>)}
          </ul>
        )}
        <p className="small muted">Low-confidence areas are shaded red in the piano roll. Your edits always override what was detected.</p>
        <div className="row">
          <button onClick={async () => { await runJob(api.analyze(proj.id, true)); await reload(); }}>Re-analyse</button>
          {perf.edited && <button className="ghost" onClick={async () => { await api.resetPerformance(proj.id); await reload(); }}>Discard my edits</button>}
          <button className="ghost" onClick={() => savePerf({ notes: perf.notes, realign_lyrics: true })}>Re-fit lyrics to notes</button>
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ Voice
export function VoiceStep({ proj, meta, save, reload, flav }: P & { flav: any }) {
  const v = proj.voice;
  const [consent, setConsent] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const engines = meta.engines.filter((e: any) => proj.input_type === "sung" || !e.input_types.every((t: string) => t === "sung"));
  const eng = meta.engines.find((e: any) => e.id === v.engine);
  const lic = eng && meta.licenses[eng.license_id];
  const rc = flav?.range;
  return (
    <div className="step-grid">
      <div>
        <h2>Voice</h2>
        <div className="seg" role="radiogroup" aria-label="Gender">
          {meta.genders.map((g: string) => (
            <button key={g} role="radio" aria-checked={v.gender === g} className={v.gender === g ? "on" : ""}
              onClick={() => save({ voice: { ...v, gender: g, voice_id: meta.voice_bank.find((b: any) => b.gender === g)?.id } })}>{g === "male" ? "Male" : "Female"}</button>
          ))}
        </div>
        <div className="seg wrap" role="radiogroup" aria-label="Age">
          {meta.ages.map((a: any) => (
            <button key={a.id} role="radio" aria-checked={v.age === a.id} className={v.age === a.id ? "on" : ""} onClick={() => save({ voice: { ...v, age: a.id } })}>{a.label}</button>
          ))}
        </div>
        <div className="voices">
          {meta.voice_bank.filter((b: any) => b.gender === v.gender).map((b: any) => (
            <button key={b.id} className={`voice ${v.voice_id === b.id ? "on" : ""}`} onClick={() => save({ voice: { ...v, voice_id: b.id } })}>
              <strong>{b.name}</strong><small>{b.desc}</small>
            </button>
          ))}
        </div>
        <p className="small muted">{meta.voice_bank_note} Age and gender change vocal-tract length, brightness, breath and vibrato habits, not just pitch.</p>
        <h3>Your own voice reference</h3>
        {proj.reference ? (
          <p className="small">Using <b>{proj.reference.original_name}</b> ({proj.reference.duration}s).
            <label className="check"><input type="checkbox" checked={!!proj.reference.use} onChange={(e) => save({ reference_use: e.target.checked })} /> Blend its timbre into the voice</label></p>
        ) : <p className="small muted">Optional. Use a recording of yourself or someone who agreed to it.</p>}
        <label className="check"><input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} /> I own this voice or have permission to use it.</label>
        <label className={`button ${consent ? "" : "disabled"}`}>Upload voice reference
          <input type="file" hidden disabled={!consent} accept=".wav,.aif,.aiff,.mp3,.m4a" onChange={async (e) => {
            const f = e.target.files?.[0]; if (!f) return; setErr(null);
            try { await api.upload(proj.id, f, f.name, "reference", consent); await reload(); } catch (x: any) { setErr(x.message); }
          }} />
        </label>
        {err && <p className="error" role="alert">{err}</p>}
      </div>
      <div>
        <h2>Natural range</h2>
        {rc ? (
          <div className={`callout ${rc.ok ? "good" : "bad"}`}>
            <p>{rc.message}</p>
            <div className="row">
              {rc.suggestions?.map((s: any) => (
                <button key={s.label} onClick={() => save({ transpose: (proj.transpose || 0) + s.semitones })}>{s.label}</button>
              ))}
            </div>
          </div>
        ) : <p className="muted">Analyse first to check the range.</p>}
        <label className="row">Transpose
          <button onClick={() => save({ transpose: (proj.transpose || 0) - 1 })} aria-label="Down a semitone">−</button>
          <b>{proj.transpose > 0 ? "+" : ""}{proj.transpose || 0} st</b>
          <button onClick={() => save({ transpose: (proj.transpose || 0) + 1 })} aria-label="Up a semitone">+</button>
          {!!proj.transpose && <button className="link" onClick={() => save({ transpose: 0 })}>Reset</button>}
        </label>
        <h3>Engine</h3>
        <select value={v.engine} onChange={(e) => save({ voice: { ...v, engine: e.target.value } })} aria-label="Engine">
          {engines.map((e: any) => <option key={e.id} value={e.id}>{e.name}{e.available ? "" : " (not installed)"}</option>)}
        </select>
        {eng && (
          <div className="small engine-info">
            <p>{eng.available ? "Ready." : eng.status} Languages: {eng.languages.join(", ")}.</p>
            {lic && <p>Licence: code {lic.code}; weights {lic.weights}; commercial use <b>{lic.commercial}</b>. {lic.notes}</p>}
            {!eng.available && lic && eng.id !== "formant" && (
              <button onClick={async () => { await api.consent(eng.license_id, true); await reload(); }}>I accept this licence and allow the download</button>
            )}
          </div>
        )}
        <label className="check"><input type="checkbox" checked={!!proj.commercial_mode} onChange={(e) => save({ commercial_mode: e.target.checked })} /> Commercial Mode (blocks export from models with unclear licences)</label>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- Flavour
export function FlavourStep({ proj, meta, save, flav }: P & { flav: any }) {
  const f = proj.flavour;
  const set = (patch: any) => save({ flavour: { ...f, ...patch } });
  const labels: Record<string, string> = { vibrato: "Vibrato", scoops: "Pitch scoops", slides: "Slides", melisma: "Melisma", breathiness: "Breathiness", grit: "Grit", timing: "Timing looseness", dynamics: "Dynamic expression", gamaka: "Gamaka (Carnatic oscillation)" };
  const disabled = new Set(f.disabled_changes || []);
  return (
    <div className="step-grid">
      <div>
        <h2>Flavour</h2>
        <label className="master">
          <span>Stay faithful</span>
          <input type="range" min={0} max={100} value={f.creative} onChange={(e) => set({ creative: +e.target.value })} aria-label="Stay faithful to creative" />
          <span>Creative</span>
          <b>{f.creative}</b>
        </label>
        <div className="seg wrap" role="radiogroup" aria-label="Style preset">
          {Object.keys(meta.presets).map((p) => (
            <button key={p} role="radio" aria-checked={f.preset === p} className={f.preset === p ? "on" : ""} onClick={() => set({ preset: p, ...meta.presets[p] })}>{p}</button>
          ))}
        </div>
        <div className="sliders">
          {meta.controls.map((c: string) => (
            <label key={c}><span>{labels[c]}</span>
              <input type="range" min={0} max={100} value={f[c] ?? 0} onChange={(e) => set({ [c]: +e.target.value, preset: "Custom" })} />
              <b>{f[c] ?? 0}</b>
            </label>
          ))}
        </div>
        <p className="small muted">Breathiness and grit change tone colour at any setting. Everything else scales with the Creative slider; at 0 the melody, timing and dynamics stay as analysed.</p>
      </div>
      <div>
        <h3>Tuning</h3>
        <div className="seg" role="radiogroup" aria-label="Tuning">
          {[["original", "Original"], ["natural", "Natural correction"], ["hard", "Hard tune"]].map(([k, l]) => (
            <button key={k} role="radio" aria-checked={f.tuning === k} className={f.tuning === k ? "on" : ""} onClick={() => set({ tuning: k })}>{l}</button>
          ))}
        </div>
        <label className="row">Scale for hard tune, harmonies and ornaments
          <select value={f.scale || ""} onChange={(e) => set({ scale: e.target.value || null })}>
            <option value="">Detected key</option>
            {meta.scales.map((s: string) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <div className="row">
          <label>Seed <input type="number" value={f.seed} disabled={f.seed_locked} onChange={(e) => set({ seed: +e.target.value })} /></label>
          <button aria-pressed={f.seed_locked} onClick={() => set({ seed_locked: !f.seed_locked })}>{f.seed_locked ? "Unlock seed" : "Lock seed"}</button>
          <button disabled={f.seed_locked} onClick={() => set({ seed: Math.floor(Math.random() * 1e6) })}>New seed</button>
        </div>
        <h3>Extra parts (each on its own stem)</h3>
        <div className="row wrap">
          <label className="check"><input type="checkbox" checked={f.doubles?.enabled} onChange={(e) => set({ doubles: { ...f.doubles, enabled: e.target.checked } })} /> Doubles</label>
          {f.doubles?.enabled && <select value={f.doubles.count} onChange={(e) => set({ doubles: { ...f.doubles, count: +e.target.value } })}>{[1, 2, 3].map((n) => <option key={n}>{n}</option>)}</select>}
          <label className="check"><input type="checkbox" checked={f.harmonies?.enabled} onChange={(e) => set({ harmonies: { ...f.harmonies, enabled: e.target.checked } })} /> Harmonies</label>
          {f.harmonies?.enabled && (
            <select value={(f.harmonies.intervals || [2]).join(",")} onChange={(e) => set({ harmonies: { ...f.harmonies, intervals: e.target.value.split(",").map(Number) } })}>
              <option value="2">Third above</option><option value="-2">Third below</option><option value="4">Fifth above</option><option value="2,4">Third + fifth above</option><option value="-3">Sixth below</option>
            </select>
          )}
          <label className="check"><input type="checkbox" checked={f.adlibs?.enabled} onChange={(e) => set({ adlibs: { enabled: e.target.checked } })} /> Ad-libs</label>
        </div>
        <h3>What the flavour engine changed</h3>
        {!flav?.changes?.length ? <p className="muted small">Nothing yet. Raise Creative or a control to add expression.</p> : (
          <ul className="changes">
            {flav.changes.map((c: any) => (
              <li key={c.id} className={c.enabled ? "" : "off"}>
                <label className="check">
                  <input type="checkbox" checked={c.enabled} onChange={() => {
                    const d = new Set(disabled);
                    d.has(c.id) ? d.delete(c.id) : d.add(c.id);
                    set({ disabled_changes: [...d] });
                  }} />
                  <span><b>{c.phrase.replace("p", "Phrase ")}</b> {c.summary}</span>
                </label>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

// ----------------------------------------------------------------- Render
export function RenderStep({ proj, runJob, job, onTime }: P & { job: Job | null; onTime: (t: number | null) => void }) {
  const [startBar, setStartBar] = useState(0);
  const [metaR, setMetaR] = useState<Record<string, any>>({});
  const [ver, setVer] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const load = async () => {
    const out: Record<string, any> = {};
    for (const m of ["preview", "full"]) { try { out[m] = await api.renderMeta(proj.id, m); } catch { /* not rendered */ } }
    setMetaR(out);
    setVer((v) => v + 1);
  };
  useEffect(() => { load(); }, [proj.id]);
  async function go(mode: "preview" | "full") {
    setErr(null);
    const j = await runJob(api.render(proj.id, mode, startBar, 8));
    if (j.status === "error") setErr(j.error);
    await load();
  }
  const busy = job && (job.status === "running" || job.status === "queued") && job.kind.startsWith("render");
  return (
    <div>
      <div className="row wrap">
        <label>Preview from bar <input type="number" min={1} value={startBar + 1} onChange={(e) => setStartBar(Math.max(0, +e.target.value - 1))} /></label>
        <button className="primary" disabled={!!busy || !proj.performance} onClick={() => go("preview")}>Render 8-bar preview</button>
        <button disabled={!!busy || !proj.performance} onClick={() => go("full")}>Render full vocal</button>
      </div>
      {err && <p className="error" role="alert">{err}</p>}
      {(["preview", "full"] as const).map((m) => metaR[m] && (
        <div key={m} className="render-block">
          <h3>{m === "preview" ? `Preview (${metaR[m].bars} bars)` : "Full vocal"}</h3>
          <p className="small muted">{metaR[m].engine.name}, seed {metaR[m].seed}, {metaR[m].duration_s}s at {metaR[m].sample_rate / 1000} kHz{metaR[m].engine.deterministic ? ", reproducible" : ", may vary between runs"}.</p>
          <Transport base={`/api/projects/${proj.id}/renders/${m}`} stems={Object.keys(metaR[m].audio_sha256)} version={ver}
            onTime={onTime} offset={metaR[m].offset_s} />
        </div>
      ))}
      <p className="small muted">Listen on headphones and switch between A and B while playing. Please judge the sound yourself: passing tests does not mean a vocal sounds good.</p>
    </div>
  );
}

// ----------------------------------------------------------------- Export
export function ExportStep({ proj, meta, runJob }: P) {
  const [o, setO] = useState({ sample_rate: 48000, bit_depth: "24", channels: 1, processed: false, bwf: false, bwf_time_reference_s: 0, midi: true });
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [rm, setRm] = useState<any>(null);
  useEffect(() => { api.renderMeta(proj.id, "full").then(setRm).catch(() => setRm(null)); }, [proj.id]);
  async function go() {
    setErr(null); setRes(null);
    const j = await runJob(api.export(proj.id, o));
    if (j.status === "error") setErr(j.error); else setRes(j.result);
  }
  const bpm = Math.round(proj.performance?.tempo?.bpm || 120);
  const key = proj.performance?.key?.short || "Cmaj";
  const age = { child: "Child", teen: "Teen", young_adult: "YoungAdult", adult: "Adult", mature: "Mature", senior: "Senior" }[proj.voice.age as string];
  const slug = (proj.song_name || "Song").replace(/\w\S*/g, (w: string) => w[0].toUpperCase() + w.slice(1)).replace(/[^A-Za-z0-9]/g, "") || "Song";
  const fn = `${slug}_LeadDry_${proj.voice.gender[0].toUpperCase() + proj.voice.gender.slice(1)}_${age}_${bpm}bpm_${key}_${o.sample_rate / 1000}k${o.bit_depth === "32f" ? "32f" : "24b"}.wav`;
  return (
    <div className="step-grid">
      <div>
        <h2>Export for Logic Pro</h2>
        {!rm && <p className="warn">Render the full vocal first.</p>}
        <div className="form">
          <label>Song name <ExportName proj={proj} /></label>
          <label>Sample rate <select value={o.sample_rate} onChange={(e) => setO({ ...o, sample_rate: +e.target.value })}>{[44100, 48000, 96000].map((s) => <option key={s} value={s}>{s / 1000} kHz</option>)}</select></label>
          <label>Format <select value={o.bit_depth} onChange={(e) => setO({ ...o, bit_depth: e.target.value })}><option value="24">24-bit PCM</option><option value="32f">32-bit float</option></select></label>
          <label>Channels <select value={o.channels} onChange={(e) => setO({ ...o, channels: +e.target.value })}><option value={1}>Mono</option><option value={2}>Stereo</option></select></label>
          <label className="check"><input type="checkbox" checked={o.processed} onChange={(e) => setO({ ...o, processed: e.target.checked })} /> Also export a processed lead (EQ, compression, room)</label>
          <label className="check"><input type="checkbox" checked={o.midi} onChange={(e) => setO({ ...o, midi: e.target.checked })} /> MIDI file with notes, lyrics and tempo map</label>
          <label className="check"><input type="checkbox" checked={o.bwf} onChange={(e) => setO({ ...o, bwf: e.target.checked })} /> BWF timestamp</label>
          {o.bwf && <label>Timestamp (seconds from project start) <input type="number" min={0} value={o.bwf_time_reference_s} onChange={(e) => setO({ ...o, bwf_time_reference_s: +e.target.value })} /></label>}
        </div>
        <p className="small mono-file">{fn}</p>
        <button className="primary" disabled={!rm} onClick={go}>Export stems</button>
        {err && <p className="error" role="alert">{err}</p>}
        {res && (
          <div className="callout good">
            <p>Exported {Object.keys(res.report.files).length} {Object.keys(res.report.files).length === 1 ? "stem" : "stems"}, {res.report.duration_s}s each, all starting at zero. Gain applied: {res.report.gain_applied_db} dB (no normalising or limiting).</p>
            <a className="button primary" href={`/api/projects/${proj.id}/exports/${res.zip}`} download>Download .zip</a>
          </div>
        )}
      </div>
      <div>
        <h2>Import into Logic</h2>
        <pre className="guide">{meta.logic_guide.split("\n").filter((l: string) => !/^=+$/.test(l)).slice(1).join("\n").trim()}</pre>
      </div>
    </div>
  );
}

function ExportName({ proj }: { proj: any }) {
  const [v, setV] = useState(proj.song_name || "");
  return <input value={v} onChange={(e) => setV(e.target.value)} onBlur={() => api.update(proj.id, { song_name: v })} />;
}

export { waitJob };
