import { useEffect, useRef, useState } from "react";
import { api, fmtTime, Job, midiName, separateApi } from "../api";
import { Recorder, startWavRecording } from "../recorder";

export type PanelProps = {
  proj: any; meta: any; save: (patch: any) => Promise<void>; reload: () => Promise<void>;
  runJob: (j: Promise<Job>) => Promise<Job>; busy: boolean;
};

const TAMIL_SAMPLE = "நிலா நிலா ஓடி வா\nநில்லாமல் ஓடி வா\nமலை மேலே ஏறி வா\nமல்லிகைப் பூ கொண்டு வா";
export const INPUT_TYPES = [
  ["hum", "Hum or melody", "Notes only. You type the lyrics."],
  ["song", "Full song with music", "Separates the vocal first, then follows it."],
  ["sung", "Sung vocal", "A clean vocal. Keeps its words and performance."],
  ["voice_sample", "Voice sample", "Only used for voice identity."],
] as const;

// ------------------------------------------------------------------ Source
export function SourcePanel({ proj, meta, save, reload, runJob, busy, analysis, savePerf }: PanelProps & { analysis: any; savePerf: (b: any) => Promise<void> }) {
  const [rec, setRec] = useState<Recorder | null>(null);
  const [upl, setUpl] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [lyrics, setLyrics] = useState(proj.lyrics);
  const [syll, setSyll] = useState<any[]>([]);
  const taps = useRef<number[]>([]);
  const [tap, setTap] = useState<number | null>(null);

  useEffect(() => setLyrics(proj.lyrics), [proj.id]);
  useEffect(() => {
    const t = setTimeout(async () => {
      if (!lyrics.text.trim()) return setSyll([]);
      try { setSyll((await api.parseLyrics(lyrics.text, lyrics.language, lyrics.dialect, lyrics.glides)).syllables); } catch { /* shown on save */ }
    }, 250);
    return () => clearTimeout(t);
  }, [lyrics]);

  async function upload(file: Blob, name: string) {
    setUpl(true); setErr(null);
    try { await api.upload(proj.id, file, name, "input"); await reload(); } catch (e: any) { setErr(e.message); } finally { setUpl(false); }
  }
  async function toggleRec() {
    if (rec) { setRec(null); upload(await rec.stop(), "recording.wav"); return; }
    try { setRec(await startWavRecording()); } catch (e: any) { setErr(`Microphone unavailable: ${e.message}`); }
  }
  const needsLyrics = proj.input_type === "hum";
  const perf = proj.performance;
  const syllCount = syll.filter((s) => !s.hold).length;

  return (
    <div className="source">
      <section className="block">
        <h2>Source</h2>
        <div className="drop" onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) upload(f, f.name); }}>
          {proj.input ? (
            <p className="src-file"><strong>{proj.input.original_name}</strong><span>{fmtTime(proj.input.duration)}, {proj.input.sample_rate / 1000} kHz, {proj.input.channels === 2 ? "stereo" : "mono"}</span></p>
          ) : <p className="dim">Drop WAV, AIFF, MP3 or M4A here. Songs up to 10 minutes are fine.</p>}
          <div className="row">
            <label className="btn">{proj.input ? "Replace" : "Choose file"}<input type="file" hidden accept=".wav,.aif,.aiff,.mp3,.m4a,.flac"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) upload(f, f.name); }} /></label>
            <button className={`btn ${rec ? "rec" : ""}`} onClick={toggleRec}>{rec ? "Stop recording" : "Record"}</button>
          </div>
          {upl && <p className="dim small">Importing…</p>}
          {err && <p className="error" role="alert">{err}</p>}
        </div>
        <div className="types" role="radiogroup" aria-label="What is this recording?">
          {INPUT_TYPES.map(([v, l, d]) => (
            <label key={v} className={`type ${proj.input_type === v ? "on" : ""}`}>
              <input type="radio" name="itype" checked={proj.input_type === v} onChange={() => save({ input_type: v })} />
              <span><strong>{l}</strong><small>{d}</small></span>
            </label>
          ))}
        </div>
      </section>

      <section className="block">
        <h2>Lyrics</h2>
        <div className="row tight">
          <select value={lyrics.language} onChange={(e) => setLyrics({ ...lyrics, language: e.target.value })} aria-label="Language">
            {Object.entries(meta.languages).map(([k, v]) => <option key={k} value={k}>{v as string}</option>)}
          </select>
          {lyrics.language !== "en" && (
            <select value={lyrics.dialect} onChange={(e) => setLyrics({ ...lyrics, dialect: e.target.value })} aria-label="Pronunciation">
              <option value="standard">ச as s (spoken, film)</option>
              <option value="literary">ச as tʃ (literary, Sri Lankan)</option>
            </select>
          )}
        </div>
        {lyrics.language !== "en" && <label className="check small"><input type="checkbox" checked={lyrics.glides} onChange={(e) => setLyrics({ ...lyrics, glides: e.target.checked })} /> Colloquial glides (எ → ye)</label>}
        <textarea className="lyrics" rows={6} value={lyrics.text} lang={lyrics.language === "ta" ? "ta" : undefined}
          placeholder={needsLyrics ? "One phrase per line. Type - to hold a syllable over extra notes." : "Optional when the recording already has words"}
          onChange={(e) => setLyrics({ ...lyrics, text: e.target.value })} onBlur={() => { if (JSON.stringify(lyrics) !== JSON.stringify(proj.lyrics)) save({ lyrics }); }} />
        {!lyrics.text && <button className="link" onClick={() => setLyrics({ ...lyrics, text: TAMIL_SAMPLE, language: "ta" })}>Use a Tamil example</button>}
        {syll.length > 0 && (
          <details className="ipa" open={syll.length < 60}>
            <summary>{syllCount} syllables{perf ? `, ${perf.notes.length} notes` : ""}. Check the pronunciation</summary>
            <div className="syllables">
              {syll.slice(0, 400).map((s, i) => (
                <span key={i} className={`syl ${s.hold ? "hold" : ""} ${s.word_start ? "ws" : ""}`} title={s.word}>
                  <b lang="ta">{s.text}</b><i>{s.hold ? "hold" : s.ipa}</i>
                </span>
              ))}
            </div>
          </details>
        )}
        <button className="btn primary wide" disabled={busy || !proj.input || (needsLyrics && !lyrics.text.trim())}
          onClick={async () => { if (JSON.stringify(lyrics) !== JSON.stringify(proj.lyrics)) await save({ lyrics }); await runJob(api.analyze(proj.id)); await reload(); }}>
          {proj.input_type === "song" && !proj.separation ? "Separate and analyse" : perf ? "Analyse again" : "Analyse"}
        </button>
        {needsLyrics && proj.input && !lyrics.text.trim() && <p className="dim small">A hummed melody needs lyrics first.</p>}
      </section>

      {analysis && perf && (
        <section className="block">
          <h2>Heard</h2>
          <div className="facts">
            <label>Tempo
              <span className="row tight">
                <input type="number" step={0.1} min={30} max={260} value={perf.tempo.bpm} onChange={(e) => { const b = +e.target.value; if (b >= 20 && b <= 400) savePerf({ tempo: { ...perf.tempo, bpm: b } }); }} aria-label="Tempo in BPM" />
                <button className="btn small" onClick={() => {
                  const now = performance.now();
                  taps.current = [...taps.current.filter((t) => now - t < 3000), now];
                  if (taps.current.length >= 3) { const d = taps.current.slice(1).map((t, i) => t - taps.current[i]); setTap(Math.round(60000 / (d.reduce((a, b) => a + b) / d.length) * 10) / 10); }
                }}>Tap</button>
                {tap && <button className="link" onClick={() => savePerf({ tempo: { ...perf.tempo, bpm: tap } })}>Use {tap}</button>}
              </span>
            </label>
            <label>Key
              <select value={`${perf.key.tonic}|${perf.key.mode}|${perf.key.tonic_pc}`} onChange={(e) => {
                const [t, m, pc] = e.target.value.split("|");
                savePerf({ key: { ...perf.key, tonic: t, mode: m, tonic_pc: +pc, label: `${t} ${m}`, short: `${t}${m === "major" ? "maj" : "min"}`, edited: true } });
              }}>
                {["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"].flatMap((t, i) => ["major", "minor"].map((m) => <option key={t + m} value={`${t}|${m}|${i}`}>{t} {m}</option>))}
              </select>
            </label>
          </div>
          <Conf v={analysis.tempo.confidence} label="Tempo" />
          <Conf v={analysis.key.confidence} label="Key" />
          <Conf v={analysis.note_confidence} label="Notes" />
          <p className="small dim">{perf.notes.length} notes in {perf.phrases.length} phrases, {midiName(Math.min(...perf.notes.map((n: any) => n.midi)))}–{midiName(Math.max(...perf.notes.map((n: any) => n.midi)))}. Analysed from {analysis.source === "separated_decoded.wav" ? "the isolated vocal" : "the original recording"}.</p>
          {analysis.warnings.map((w: string, i: number) => <p key={i} className="warn small">{w}</p>)}
          {analysis.low_confidence.length > 0 && <p className="small"><span className="warn">{analysis.low_confidence.length} unsure areas</span> are shaded red on the notes lane.</p>}
          <div className="row tight">
            {perf.edited && <button className="btn small ghost" onClick={async () => { await api.resetPerformance(proj.id); await reload(); }}>Discard my note edits</button>}
            <button className="btn small ghost" onClick={() => savePerf({ notes: perf.notes, realign_lyrics: true })}>Re-fit lyrics</button>
          </div>
        </section>
      )}
    </div>
  );
}

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

// --------------------------------------------------------------- Separate
export function SeparatePanel({ proj, meta, save, reload, runJob, busy }: PanelProps) {
  const [method, setMethod] = useState(proj.separation_method || "builtin");
  const [strength, setStrength] = useState(1);
  const [fmt, setFmt] = useState({ sample_rate: 48000, bit_depth: "24", channels: "original" as any });
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const sep = proj.separation;
  const m = meta.separation[method];
  async function run() {
    setErr(null);
    const j = await runJob(separateApi.run(proj.id, method, strength));
    if (j.status === "error") setErr(j.error);
    await reload();
  }
  async function exp(stems: string[]) {
    setErr(null); setRes(null);
    const j = await runJob(separateApi.export(proj.id, { stems, ...fmt }));
    if (j.status === "error") setErr(j.error); else setRes(j.result);
  }
  return (
    <div>
      <p className="lede">Split a finished song into the singer's voice and the music. Use the isolated vocal to follow the melody, or export either part on its own.</p>
      <div className="methods" role="radiogroup" aria-label="Separation method">
        {Object.entries(meta.separation).map(([k, v]: any) => (
          <label key={k} className={`type ${method === k ? "on" : ""}`}>
            <input type="radio" name="sepm" checked={method === k} onChange={() => setMethod(k)} />
            <span><strong>{v.name}</strong><small>{v.notes}{v.download_mb ? ` About ${v.download_mb} MB, ${v.license.code} licence.` : ""}</small>
              {!v.available && <small className="warn">{v.status}</small>}</span>
          </label>
        ))}
      </div>
      {!m.available && method === "demucs" && (
        <button className="btn small" onClick={async () => { await api.consent("demucs", true); await reload(); }}>I accept the MIT licence and allow the download</button>
      )}
      {method === "builtin" && (
        <label className="slider"><span>Isolation</span>
          <input type="range" min={0.5} max={2} step={0.1} value={strength} onChange={(e) => setStrength(+e.target.value)} />
          <b>{strength < 0.9 ? "Gentle" : strength > 1.2 ? "Strong" : "Balanced"}</b>
        </label>
      )}
      <button className="btn primary wide" disabled={busy || !proj.input || !m.available} onClick={run}>
        {sep ? "Separate again" : "Separate vocals"}
      </button>
      {!proj.input && <p className="dim small">Import a song first.</p>}
      {sep && (
        <div className="result">
          <p><strong>Done</strong> with {meta.separation[sep.method]?.name ?? sep.method} in {sep.runtime_s}s for {fmtTime(sep.duration_s)} of audio.
            The isolated vocal and instrumental are now lanes on the timeline. Solo them to listen.</p>
          {sep.method === "builtin" && <p className="small dim">The quick method can leave some music in the vocal on dense mixes. Studio mode (Demucs) is cleaner.</p>}
          <h3>Export</h3>
          <div className="grid2">
            <label>Sample rate<select value={fmt.sample_rate} onChange={(e) => setFmt({ ...fmt, sample_rate: +e.target.value })}>{[44100, 48000, 96000].map((s) => <option key={s} value={s}>{s / 1000} kHz</option>)}</select></label>
            <label>Format<select value={fmt.bit_depth} onChange={(e) => setFmt({ ...fmt, bit_depth: e.target.value })}><option value="24">24-bit PCM</option><option value="32f">32-bit float</option></select></label>
            <label>Channels<select value={String(fmt.channels)} onChange={(e) => setFmt({ ...fmt, channels: e.target.value === "original" ? "original" : +e.target.value })}><option value="original">As the song</option><option value="1">Mono</option><option value="2">Stereo</option></select></label>
          </div>
          <div className="row">
            <button className="btn primary" disabled={busy} onClick={() => exp(["vocals"])} style={{ ["--btn" as any]: "var(--vocal)" }}>Export vocals only</button>
            <button className="btn" disabled={busy} onClick={() => exp(["instrumental"])}>Export instrumental</button>
            <button className="btn ghost" disabled={busy} onClick={() => exp(["vocals", "instrumental"])}>Both (.zip)</button>
          </div>
          {res && <p className="ok-line">Ready: <a href={`/api/projects/${proj.id}/exports/${res.file}`} download>{res.file}</a>{res.report.gain_applied_db < 0 ? ` (gain ${res.report.gain_applied_db} dB to avoid clipping)` : ""}</p>}
        </div>
      )}
      {err && <p className="error" role="alert">{err}</p>}
    </div>
  );
}

// ------------------------------------------------------------------ Voice
export function VoicePanel({ proj, meta, save, reload, flav }: PanelProps & { flav: any }) {
  const v = proj.voice;
  const [consent, setConsent] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const engines = meta.engines.filter((e: any) => ["sung", "song"].includes(proj.input_type) || !e.input_types.every((t: string) => t === "sung"));
  const eng = meta.engines.find((e: any) => e.id === v.engine);
  const lic = eng && meta.licenses[eng.license_id];
  const rc = flav?.range;
  return (
    <div>
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
      <h3>Range</h3>
      {rc ? (
        <div className={`note-box ${rc.ok ? "good" : "bad"}`}>
          <p>{rc.message}</p>
          <div className="row tight">{rc.suggestions?.map((s: any) => <button key={s.label} className="btn small" onClick={() => save({ transpose: (proj.transpose || 0) + s.semitones })}>{s.label}</button>)}</div>
        </div>
      ) : <p className="dim small">Analyse first to check the range.</p>}
      <div className="row tight">Transpose
        <button className="icon" onClick={() => save({ transpose: (proj.transpose || 0) - 1 })} aria-label="Down a semitone">−</button>
        <b className="num">{proj.transpose > 0 ? "+" : ""}{proj.transpose || 0}</b>
        <button className="icon" onClick={() => save({ transpose: (proj.transpose || 0) + 1 })} aria-label="Up a semitone">+</button>
        {!!proj.transpose && <button className="link" onClick={() => save({ transpose: 0 })}>Reset</button>}
      </div>
      <h3>Engine</h3>
      <select value={v.engine} onChange={(e) => save({ voice: { ...v, engine: e.target.value } })} aria-label="Engine">
        {engines.map((e: any) => <option key={e.id} value={e.id}>{e.name}{e.available ? "" : " (not installed)"}</option>)}
      </select>
      {eng && (
        <div className="small engine-info">
          <p>{eng.available ? "Ready." : eng.status} Sings: {eng.languages.join(", ")}.</p>
          {lic && <p className="dim">Code {lic.code}. Weights {lic.weights}. Commercial use: <b>{lic.commercial}</b>.</p>}
          {!eng.available && lic && eng.id !== "formant" && <button className="btn small" onClick={async () => { await api.consent(eng.license_id, true); await reload(); }}>Accept licence and allow download</button>}
        </div>
      )}
      <h3>Your own voice</h3>
      {proj.reference ? (
        <label className="check small"><input type="checkbox" checked={!!proj.reference.use} onChange={(e) => save({ reference_use: e.target.checked })} /> Blend in {proj.reference.original_name}</label>
      ) : <p className="dim small">Optional. A recording of yourself, or of someone who agreed.</p>}
      <label className="check small"><input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} /> I own this voice or have permission to use it.</label>
      <label className={`btn small ${consent ? "" : "disabled"}`}>Upload voice reference
        <input type="file" hidden disabled={!consent} accept=".wav,.aif,.aiff,.mp3,.m4a" onChange={async (e) => {
          const f = e.target.files?.[0]; if (!f) return; setErr(null);
          try { await api.upload(proj.id, f, f.name, "reference", consent); await reload(); } catch (x: any) { setErr(x.message); }
        }} />
      </label>
      {err && <p className="error" role="alert">{err}</p>}
      <label className="check small commercial"><input type="checkbox" checked={!!proj.commercial_mode} onChange={(e) => save({ commercial_mode: e.target.checked })} /> Commercial Mode: block exports from models with unclear licences</label>
    </div>
  );
}

// ---------------------------------------------------------------- Flavour
const LABELS: Record<string, string> = { vibrato: "Vibrato", scoops: "Pitch scoops", slides: "Slides", melisma: "Melisma", breathiness: "Breathiness", grit: "Grit", timing: "Timing looseness", dynamics: "Dynamics", gamaka: "Gamaka" };

export function FlavourPanel({ proj, meta, save, flav }: PanelProps & { flav: any }) {
  const f = proj.flavour;
  const set = (patch: any) => save({ flavour: { ...f, ...patch } });
  const disabled = new Set(f.disabled_changes || []);
  return (
    <div>
      <label className="master">
        <span>Faithful</span>
        <input type="range" min={0} max={100} value={f.creative} onChange={(e) => set({ creative: +e.target.value })} aria-label="Faithful to creative" />
        <span>Creative</span>
        <b className="num">{f.creative}</b>
      </label>
      <div className="seg wrap" role="radiogroup" aria-label="Style">
        {Object.keys(meta.presets).map((p) => (
          <button key={p} role="radio" aria-checked={f.preset === p} className={f.preset === p ? "on" : ""} onClick={() => set({ preset: p, ...meta.presets[p] })}>{p === "Indian Classical-inspired" ? "Carnatic-inspired" : p}</button>
        ))}
      </div>
      <div className="sliders">
        {meta.controls.map((c: string) => (
          <label key={c} className="slider"><span>{LABELS[c]}</span>
            <input type="range" min={0} max={100} value={f[c] ?? 0} onChange={(e) => set({ [c]: +e.target.value, preset: "Custom" })} />
            <b className="num">{f[c] ?? 0}</b>
          </label>
        ))}
      </div>
      <p className="small dim">At Faithful, notes, timing and dynamics stay as analysed. Breathiness and grit only colour the tone.</p>
      <h3>Tuning</h3>
      <div className="seg" role="radiogroup" aria-label="Tuning">
        {[["original", "Original"], ["natural", "Natural"], ["hard", "Hard tune"]].map(([k, l]) => (
          <button key={k} role="radio" aria-checked={f.tuning === k} className={f.tuning === k ? "on" : ""} onClick={() => set({ tuning: k })}>{l}</button>
        ))}
      </div>
      <label className="stack">Scale or raga
        <select value={f.scale || ""} onChange={(e) => set({ scale: e.target.value || null })}>
          <option value="">Detected key</option>
          {meta.scales.map((s: string) => <option key={s} value={s}>{s}</option>)}
        </select>
      </label>
      <div className="row tight">
        <label className="stack">Seed <input type="number" value={f.seed} disabled={f.seed_locked} onChange={(e) => set({ seed: +e.target.value })} /></label>
        <button className="btn small" aria-pressed={f.seed_locked} onClick={() => set({ seed_locked: !f.seed_locked })}>{f.seed_locked ? "Unlock" : "Lock"}</button>
        <button className="btn small" disabled={f.seed_locked} onClick={() => set({ seed: Math.floor(Math.random() * 1e6) })}>New seed</button>
      </div>
      <h3>Extra parts</h3>
      <div className="parts">
        <label className="check" style={{ ["--lane" as any]: "var(--double)" }}><input type="checkbox" checked={f.doubles?.enabled} onChange={(e) => set({ doubles: { ...f.doubles, enabled: e.target.checked } })} /><span className="dot" />Doubles
          {f.doubles?.enabled && <select value={f.doubles.count} onChange={(e) => set({ doubles: { ...f.doubles, count: +e.target.value } })} aria-label="Number of doubles">{[1, 2, 3].map((n) => <option key={n}>{n}</option>)}</select>}</label>
        <label className="check" style={{ ["--lane" as any]: "var(--harmony)" }}><input type="checkbox" checked={f.harmonies?.enabled} onChange={(e) => set({ harmonies: { ...f.harmonies, enabled: e.target.checked } })} /><span className="dot" />Harmonies
          {f.harmonies?.enabled && (
            <select value={(f.harmonies.intervals || [2]).join(",")} onChange={(e) => set({ harmonies: { ...f.harmonies, intervals: e.target.value.split(",").map(Number) } })} aria-label="Harmony interval">
              <option value="2">Third above</option><option value="-2">Third below</option><option value="4">Fifth above</option><option value="2,4">Third + fifth</option><option value="-3">Sixth below</option>
            </select>
          )}</label>
        <label className="check" style={{ ["--lane" as any]: "var(--adlib)" }}><input type="checkbox" checked={f.adlibs?.enabled} onChange={(e) => set({ adlibs: { enabled: e.target.checked } })} /><span className="dot" />Ad-libs</label>
      </div>
      <h3>Changes made</h3>
      {!flav?.changes?.length ? <p className="dim small">None yet. Move toward Creative to add expression.</p> : (
        <ul className="changes">
          {flav.changes.map((c: any) => (
            <li key={c.id} className={c.enabled ? "" : "off"}>
              <label className="check small">
                <input type="checkbox" checked={c.enabled} onChange={() => { const d = new Set(disabled); d.has(c.id) ? d.delete(c.id) : d.add(c.id); set({ disabled_changes: [...d] }); }} />
                <span><b>{c.phrase.replace("p", "Phrase ")}</b> {c.summary}</span>
              </label>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// ----------------------------------------------------------------- Export
export function ExportPanel({ proj, meta, runJob, busy, renderMeta }: PanelProps & { renderMeta: any }) {
  const [o, setO] = useState({ sample_rate: 48000, bit_depth: "24", channels: 1, processed: false, bwf: false, bwf_time_reference_s: 0, midi: true, include_instrumental: !!proj.separation });
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [name, setName] = useState(proj.song_name || "");
  async function go() {
    setErr(null); setRes(null);
    await api.update(proj.id, { song_name: name });
    const j = await runJob(api.export(proj.id, o));
    if (j.status === "error") setErr(j.error); else setRes(j.result);
  }
  const bpm = Math.round(proj.performance?.tempo?.bpm || 120);
  const key = proj.performance?.key?.short || "Cmaj";
  const age = ({ child: "Child", teen: "Teen", young_adult: "YoungAdult", adult: "Adult", mature: "Mature", senior: "Senior" } as any)[proj.voice.age];
  const slug = (name || "Song").replace(/\w\S*/g, (w: string) => w[0].toUpperCase() + w.slice(1)).replace(/[^A-Za-z0-9]/g, "") || "Song";
  const g = proj.voice.gender[0].toUpperCase() + proj.voice.gender.slice(1);
  const fn = `${slug}_LeadDry_${g}_${age}_${bpm}bpm_${key}_${o.sample_rate / 1000}k${o.bit_depth === "32f" ? "32f" : "24b"}.wav`;
  return (
    <div>
      <p className="lede">Stems for Logic Pro. Every file starts at bar 1 and has the same length.</p>
      {!renderMeta && <p className="warn small">Render the full song first (top bar).</p>}
      <label className="stack">Song name <input value={name} onChange={(e) => setName(e.target.value)} /></label>
      <div className="grid2">
        <label>Sample rate<select value={o.sample_rate} onChange={(e) => setO({ ...o, sample_rate: +e.target.value })}>{[44100, 48000, 96000].map((s) => <option key={s} value={s}>{s / 1000} kHz</option>)}</select></label>
        <label>Format<select value={o.bit_depth} onChange={(e) => setO({ ...o, bit_depth: e.target.value })}><option value="24">24-bit PCM</option><option value="32f">32-bit float</option></select></label>
        <label>Channels<select value={o.channels} onChange={(e) => setO({ ...o, channels: +e.target.value })}><option value={1}>Mono</option><option value={2}>Stereo</option></select></label>
      </div>
      <label className="check small"><input type="checkbox" checked={o.processed} onChange={(e) => setO({ ...o, processed: e.target.checked })} /> Add a processed lead (EQ, compression, room)</label>
      <label className="check small"><input type="checkbox" checked={o.include_instrumental} disabled={!proj.separation} onChange={(e) => setO({ ...o, include_instrumental: e.target.checked })} /> Include the instrumental from separation</label>
      <label className="check small"><input type="checkbox" checked={o.midi} onChange={(e) => setO({ ...o, midi: e.target.checked })} /> MIDI with notes, lyrics and tempo map</label>
      <label className="check small"><input type="checkbox" checked={o.bwf} onChange={(e) => setO({ ...o, bwf: e.target.checked })} /> BWF timestamp</label>
      {o.bwf && <label className="stack">Timestamp, seconds from project start <input type="number" min={0} value={o.bwf_time_reference_s} onChange={(e) => setO({ ...o, bwf_time_reference_s: +e.target.value })} /></label>}
      <p className="filename">{fn}</p>
      <button className="btn primary wide" disabled={busy || !renderMeta} onClick={go}>Export stems</button>
      {err && <p className="error" role="alert">{err}</p>}
      {res && (
        <div className="note-box good">
          <p>{Object.keys(res.report.files).length} {Object.keys(res.report.files).length === 1 ? "file" : "files"}, {fmtTime(res.report.duration_s)} each. Gain {res.report.gain_applied_db} dB. Nothing normalised or limited.</p>
          <a className="btn primary" href={`/api/projects/${proj.id}/exports/${res.zip}`} download>Download .zip</a>
        </div>
      )}
      <details className="guide">
        <summary>How to import into Logic</summary>
        <pre>{meta.logic_guide.split("\n").filter((l: string) => !/^=+$/.test(l)).slice(1).join("\n").trim()}</pre>
      </details>
    </div>
  );
}
