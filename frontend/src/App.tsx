import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, fmtTime, Job, Note, playUrl, waitJob } from "./api";
import { ExportPanel, FlavourPanel, SeparatePanel, SourcePanel, VoicePanel } from "./components/Panels";
import PianoRoll from "./components/PianoRoll";
import Timeline, { Lane } from "./components/Timeline";
import { useMixer } from "./useMixer";

const TABS = [
  { id: "separate", label: "Separate" },
  { id: "voice", label: "Voice" },
  { id: "flavour", label: "Flavour" },
  { id: "export", label: "Export" },
] as const;
type Compare = "original" | "vocal" | "new" | "newmusic";

export default function App() {
  const [meta, setMeta] = useState<any>(null);
  const [list, setList] = useState<any[]>([]);
  const [proj, setProj] = useState<any>(null);
  const [analysis, setAnalysis] = useState<any>(null);
  const [flav, setFlav] = useState<any>(null);
  const [renders, setRenders] = useState<Record<string, any>>({});
  const [ver, setVer] = useState(0);
  const [tab, setTab] = useState<string>("voice");
  const [job, setJob] = useState<Job | null>(null);
  const [showLog, setShowLog] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [view, setView] = useState({ pxs: 40, scrollX: 0 });
  const [mute, setMute] = useState<Record<string, boolean>>({});
  const [solo, setSolo] = useState<string | null>(null);
  const [compare, setCompare] = useState<Compare>("original");
  const [showRender, setShowRender] = useState<"full" | "preview">("full");
  const perfTimer = useRef<number>(0);

  useEffect(() => {
    api.meta().then(setMeta).catch((e) => setErr(`Cannot reach the local Kuralkoodu server (${e.message}). Start it with ./run.sh`));
    api.projects().then(setList).catch(() => {});
  }, []);

  const loadRenders = useCallback(async (id: string) => {
    const out: Record<string, any> = {};
    for (const m of ["preview", "full"]) { try { out[m] = await api.renderMeta(id, m); } catch { /* none yet */ } }
    setRenders(out);
    setVer((v) => v + 1);
    return out;
  }, []);

  const loadProject = useCallback(async (id: string) => {
    const p = await api.project(id);
    setProj(p);
    try { localStorage.setItem("vf:last", id); } catch { /* private mode */ }
    try { setAnalysis(await api.analysis(id)); } catch { setAnalysis(null); }
    if (p.performance) { try { setFlav(await api.flavour(id)); } catch { setFlav(null); } } else setFlav(null);
    setMeta(await api.meta());
    await loadRenders(id);
    return p;
  }, [loadRenders]);

  useEffect(() => {
    let last: string | null = null;
    try { last = localStorage.getItem("vf:last"); } catch { /* private mode */ }
    if (last) loadProject(last).catch(() => {});
  }, [loadProject]);

  const reload = async () => { if (proj) { await loadProject(proj.id); setVer((v) => v + 1); } };
  const save = async (patch: any) => {
    const p = await api.update(proj.id, patch);
    setProj(p);
    if (p.performance) setFlav(await api.flavour(proj.id));
  };
  const savePerf = async (body: any) => {
    const p = await api.performance(proj.id, body);
    setProj(p);
    setFlav(await api.flavour(proj.id));
  };
  const runJob = async (pj: Promise<Job>) => {
    setErr(null);
    try {
      const j = await waitJob(await pj, setJob);
      if (j.status === "error") setErr(j.error);
      return j;
    } catch (e: any) { setErr(e.message); throw e; }
  };
  const busy = !!job && (job.status === "running" || job.status === "queued");
  const onNotes = (notes: Note[]) => {
    setProj((p: any) => ({ ...p, performance: { ...p.performance, notes } }));
    window.clearTimeout(perfTimer.current);
    perfTimer.current = window.setTimeout(() => savePerf({ notes }), 350);
  };

  async function newProject(kind: "hum" | "song") {
    const name = window.prompt("Song name", kind === "song" ? "My song" : "Untitled") ?? "";
    if (!name) return;
    const p = await api.create(name);
    await api.update(p.id, { input_type: kind });
    setList(await api.projects());
    await loadProject(p.id);
    setTab(kind === "song" ? "separate" : "voice");
  }

  // ---------------- lanes on the shared timeline
  const rm = renders[showRender] || renders.full || renders.preview;
  const rmMode = renders[showRender] ? showRender : renders.full ? "full" : "preview";
  const lanes: Lane[] = useMemo(() => {
    if (!proj?.input) return [];
    const L: Lane[] = [{ id: "input", label: proj.input_type === "song" ? "Song" : "Your recording", color: "--mix", src: "input", offset: 0, version: ver, note: proj.input.original_name }];
    if (proj.separation) {
      L.push({ id: "vocals", label: "Isolated vocal", color: "--vocal", src: "vocals", offset: 0, version: ver, note: proj.separation.method === "demucs" ? "Demucs" : "Quick separation" });
      L.push({ id: "instrumental", label: "Instrumental", color: "--music", src: "instrumental", offset: 0, version: ver, note: "Music without the voice" });
    }
    if (rm) {
      const label: Record<string, [string, string]> = { lead: ["New lead", "--lead"], doubles: ["Doubles", "--double"], harmonies: ["Harmonies", "--harmony"], adlibs: ["Ad-libs", "--adlib"] };
      for (const s of Object.keys(rm.audio_sha256)) {
        const [lab, col] = label[s] || [s, "--lead"];
        L.push({ id: `r-${s}`, label: lab, color: col, src: `render:${rmMode}:${s}`, offset: rm.offset_s || 0, version: ver,
          note: s === "lead" ? `${rmMode === "preview" ? `Preview, ${rm.bars} bars` : "Full song"}, seed ${rm.seed}` : undefined,
          download: { href: `/api/projects/${proj.id}/renders/${rmMode}/${s}.wav`, label: `Download ${lab} (32-bit float WAV)` } });
      }
    }
    return L;
  }, [proj?.input, proj?.separation, proj?.input_type, rm, rmMode, ver]);

  const compareSets: Record<Compare, (id: string) => boolean> = {
    original: (id) => id === "input",
    vocal: (id) => id === "vocals",
    new: (id) => id.startsWith("r-"),
    newmusic: (id) => id.startsWith("r-") || id === "instrumental",
  };
  const audible = (id: string) => (solo ? solo === id : compareSets[compare](id) && !mute[id]);
  const mixLanes = lanes.map((l) => ({ id: l.id, url: playUrl(proj.id, l.src, l.version), offset: l.offset, audible: audible(l.id) }));
  const mixer = useMixer(mixLanes);

  const duration = Math.max(proj?.input?.duration || 0, analysis?.duration || 0, ...(Object.values(renders) as any[]).map((r) => (r.offset_s || 0) + r.duration_s), 4);
  const bpm = proj?.performance?.tempo?.bpm || 120;
  const beatOffset = proj?.performance?.tempo?.beat_offset || 0;

  useEffect(() => { // fit the song when a project opens
    if (!proj?.input) return;
    const w = Math.max(300, window.innerWidth - (window.innerWidth > 1100 ? 300 + 380 + 200 : 200));
    setView({ pxs: Math.max(2, w / Math.max(1, proj.input.duration)), scrollX: 0 });
  }, [proj?.id, proj?.input?.hash]);

  useEffect(() => { // keep the playhead in view while playing
    if (!mixer.playing) return;
    const visible = (window.innerWidth * 0.5) / view.pxs;
    if (mixer.pos > view.scrollX + visible * 1.6 || mixer.pos < view.scrollX) setView((v) => ({ ...v, scrollX: Math.max(0, mixer.pos - visible * 0.2) }));
  }, [mixer.pos]);

  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (e.code === "Space" && !["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(t.tagName)) {
        e.preventDefault();
        mixer.playing ? mixer.pause() : mixer.play();
      }
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [mixer.playing, mixer.play, mixer.pause]);

  async function render(mode: "preview" | "full") {
    const bar = (4 * 60) / bpm;
    const startBar = Math.max(0, Math.floor((mixer.pos - beatOffset) / bar));
    const j = await runJob(api.render(proj.id, mode, startBar, 8));
    if (j.status === "done") {
      await loadRenders(proj.id);
      setShowRender(mode);
      setCompare(proj.separation ? "newmusic" : "new");
    }
  }

  if (!meta) return <div className="boot">{err ? <p className="error" role="alert">{err}</p> : <p>Starting Kuralkoodu…</p>}</div>;

  const props = { proj, meta, save, reload, runJob, busy };
  const hasSep = !!proj?.separation;
  const hasRender = !!rm;

  return (
    <div className="app">
      <header className="bar">
        <div className="brand"><span className="glyph" lang="ta">குரல்கூடு</span><span>Kuralkoodu</span></div>
        <select className="song-pick" value={proj?.id || ""} onChange={(e) => e.target.value && loadProject(e.target.value)} aria-label="Open a song">
          <option value="">Open a song…</option>
          {list.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select>
        <button className="btn ghost small" onClick={() => newProject("hum")}>New song</button>

        {proj?.input && (
          <div className="transport" role="group" aria-label="Transport">
            <button className="play" onClick={() => (mixer.playing ? mixer.pause() : mixer.play())} aria-label={mixer.playing ? "Pause" : "Play"} disabled={mixer.loading}>
              {mixer.loading ? "…" : mixer.playing ? "❚❚" : "▶"}
            </button>
            <button className="icon" onClick={() => { mixer.stop(); mixer.seek(0); }} aria-label="Back to start">⏮</button>
            <span className="clock num">{fmtTime(mixer.pos)}<small> / {fmtTime(duration)}</small></span>
            <div className="compare" role="radiogroup" aria-label="Listen to">
              {([["original", "Original", true], ["vocal", "Vocal only", hasSep], ["new", "New vocal", hasRender], ["newmusic", "New + music", hasRender && hasSep]] as [Compare, string, boolean][])
                .filter(([, , ok]) => ok).map(([k, l]) => (
                  <button key={k} role="radio" aria-checked={compare === k} className={compare === k ? "on" : ""} onClick={() => { setCompare(k); setSolo(null); }}>{l}</button>
                ))}
            </div>
          </div>
        )}
        {proj?.performance && (
          <div className="render-btns">
            <button className="btn small" disabled={busy} onClick={() => render("preview")} title="8 bars starting at the playhead">Preview 8 bars</button>
            <button className="btn primary small" disabled={busy} onClick={() => render("full")}>Render full song</button>
          </div>
        )}
        {job && (
          <button className={`job-pill ${job.status}`} onClick={() => setShowLog(!showLog)} aria-expanded={showLog}>
            <span>{busy ? job.message || job.kind : `${job.kind.replace("-", " ")}: ${job.status}`}</span>
            <progress max={1} value={job.progress} />
          </button>
        )}
        {busy && <button className="btn small danger" onClick={() => api.cancel(job!.id)}>Cancel</button>}
      </header>
      {mixer.error && <p className="error banner" role="alert">{mixer.error}</p>}
      {err && <p className="error banner" role="alert">{err} <button className="link" onClick={() => setErr(null)}>Dismiss</button></p>}
      {showLog && job && <pre className="log">{job.logs.join("\n")}</pre>}

      {!proj ? (
        <main className="welcome">
          <h1>Hum it, or bring a song. Leave with a lead vocal.</h1>
          <p>Kuralkoodu follows your melody, sings your Tamil or English lyrics on it, separates vocals from finished songs, and hands Logic Pro aligned stems. Everything stays on this computer.</p>
          <div className="starts">
            <button className="start" onClick={() => newProject("hum")} style={{ ["--lane" as any]: "var(--lead)" }}>
              <strong>Make a vocal from a melody</strong><span>Hum, sing or play the tune, then type the words.</span>
            </button>
            <button className="start" onClick={() => newProject("song")} style={{ ["--lane" as any]: "var(--vocal)" }}>
              <strong>Separate vocals from a song</strong><span>Split a finished track into voice and music, and export either part.</span>
            </button>
          </div>
        </main>
      ) : (
        <div className="studio">
          <aside className="left">
            <SourcePanel {...props} analysis={analysis} savePerf={savePerf} />
          </aside>
          <main className="center">
            <div className="song-title">
              <h1>{proj.name}</h1>
              {renders.full && renders.preview && (
                <div className="seg small" role="radiogroup" aria-label="Render shown">
                  {(["preview", "full"] as const).map((m) => <button key={m} role="radio" aria-checked={rmMode === m} className={rmMode === m ? "on" : ""} onClick={() => setShowRender(m)}>{m === "full" ? "Full render" : "Preview"}</button>)}
                </div>
              )}
            </div>
            {!proj.input ? (
              <div className="empty"><p>Add audio on the left to start: a hummed tune, a sung line, or a full song.</p></div>
            ) : (
              <Timeline pid={proj.id} lanes={lanes} view={view} onView={setView} duration={duration} bpm={bpm} beatOffset={beatOffset}
                playhead={mixer.pos} onSeek={(t) => mixer.seek(t)} mute={mute} solo={solo}
                onMute={(id) => setMute((m) => ({ ...m, [id]: !m[id] }))} onSolo={(id) => setSolo((s) => (s === id ? null : id))}
                rollLabel={<><strong>Notes and pitch</strong><small>{proj.performance ? `${proj.performance.notes.length} notes, ${Math.round(bpm * 10) / 10} BPM, ${proj.performance.key.label}` : "Analyse to see notes"}</small></>}
                roll={proj.performance ? (
                  <PianoRoll notes={proj.performance.notes} analysis={analysis} bpm={bpm} beatOffset={beatOffset} flavouredNotes={flav?.notes}
                    playhead={mixer.pos} onChange={onNotes} pxs={view.pxs} scrollX={view.scrollX}
                    onView={(pxs, scrollX) => setView({ pxs: Math.min(600, Math.max(2, pxs)), scrollX })} onSeek={(t) => mixer.seek(t)} />
                ) : undefined} />
            )}
            <p className="hint">Space plays and pauses. Click a lane to move the playhead. Ctrl or ⌘ + scroll zooms. M mutes a lane, S solos it. Judge the sound with your ears: passing tests does not mean it sounds good.</p>
          </main>
          <aside className="right">
            <nav className="tabs" role="tablist" aria-label="Inspector">
              {TABS.map((t) => (
                <button key={t.id} role="tab" aria-selected={tab === t.id} className={tab === t.id ? "on" : ""} onClick={() => setTab(t.id)}>
                  {t.label}{t.id === "separate" && hasSep && <span className="tick" aria-label="done"> ✓</span>}
                </button>
              ))}
            </nav>
            <div className="tab-body" role="tabpanel">
              {tab === "separate" && <SeparatePanel {...props} />}
              {tab === "voice" && <VoicePanel {...props} flav={flav} />}
              {tab === "flavour" && <FlavourPanel {...props} flav={flav} />}
              {tab === "export" && <ExportPanel {...props} renderMeta={renders.full} />}
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}
