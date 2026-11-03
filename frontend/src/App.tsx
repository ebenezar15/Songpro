import { useCallback, useEffect, useRef, useState } from "react";
import { api, Job, Note, waitJob } from "./api";
import PianoRoll from "./components/PianoRoll";
import { AnalysisStep, ExportStep, FlavourStep, ImportStep, RenderStep, VoiceStep } from "./components/Steps";

const STEPS = [
  { id: "import", label: "Import", hint: "Audio, input type, lyrics" },
  { id: "review", label: "Review", hint: "Tempo, key, notes" },
  { id: "voice", label: "Voice", hint: "Gender, age, range" },
  { id: "flavour", label: "Flavour", hint: "Style and expression" },
  { id: "render", label: "Render", hint: "Preview, A/B" },
  { id: "export", label: "Export", hint: "Stems for Logic" },
];

export default function App() {
  const [meta, setMeta] = useState<any>(null);
  const [list, setList] = useState<any[]>([]);
  const [proj, setProj] = useState<any>(null);
  const [analysis, setAnalysis] = useState<any>(null);
  const [flav, setFlav] = useState<any>(null);
  const [step, setStep] = useState("import");
  const [job, setJob] = useState<Job | null>(null);
  const [showLogs, setShowLogs] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [playhead, setPlayhead] = useState<number | null>(null);
  const perfTimer = useRef<number>(0);

  useEffect(() => {
    api.meta().then(setMeta).catch((e) => setErr(`Cannot reach the local VocalForge server: ${e.message}`));
    api.projects().then(setList).catch(() => {});
  }, []);

  const loadProject = useCallback(async (id: string) => {
    const p = await api.project(id);
    setProj(p);
    localStorage.setItem("vf:last", id);
    try { setAnalysis(await api.analysis(id)); } catch { setAnalysis(null); }
    if (p.performance) { try { setFlav(await api.flavour(id)); } catch { setFlav(null); } } else setFlav(null);
    setMeta(await api.meta());
    return p;
  }, []);

  useEffect(() => {
    let last: string | null = null;
    try { last = localStorage.getItem("vf:last"); } catch { /* private mode */ }
    if (last) loadProject(last).then((p) => setStep(p.performance ? "review" : "import")).catch(() => {});
  }, [loadProject]);

  const reload = async () => { if (proj) await loadProject(proj.id); };
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
  const onNotes = (notes: Note[]) => {
    setProj((p: any) => ({ ...p, performance: { ...p.performance, notes } }));
    window.clearTimeout(perfTimer.current);
    perfTimer.current = window.setTimeout(() => savePerf({ notes }), 350);
  };

  async function newProject() {
    const name = window.prompt("Song name", "Untitled") ?? "";
    if (!name) return;
    const p = await api.create(name);
    setList(await api.projects());
    await loadProject(p.id);
    setStep("import");
  }

  if (!meta) return <div className="boot">{err ? <p className="error" role="alert">{err}<br />Start it with <code>./run.sh</code>.</p> : <p>Starting VocalForge…</p>}</div>;

  const props = { proj, meta, save, reload, runJob };
  const stepIndex = STEPS.findIndex((s) => s.id === step);
  const done: Record<string, boolean> = {
    import: !!proj?.input, review: !!proj?.performance, voice: !!proj?.performance, flavour: !!proj?.performance,
    render: false, export: false,
  };

  return (
    <div className="app">
      <header className="top">
        <div className="mark" aria-label="VocalForge"><span className="glyph" aria-hidden>ஸ்வ</span> VocalForge</div>
        <select value={proj?.id || ""} onChange={(e) => e.target.value && loadProject(e.target.value)} aria-label="Project">
          <option value="">Open a project…</option>
          {list.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select>
        <button onClick={newProject}>New song</button>
        <span className="local">Runs on this computer only. No audio leaves it.</span>
      </header>

      {!proj ? (
        <main className="welcome">
          <h1>Hum it. Type the words. Get a lead vocal.</h1>
          <p>VocalForge reads the melody you hum or sing, sings your lyrics on it in Tamil or English, and hands Logic Pro aligned WAV stems, MIDI and a session file.</p>
          <button className="primary big" onClick={newProject}>Start a new song</button>
        </main>
      ) : (
        <div className="body">
          <nav className="steps" aria-label="Workflow">
            <ol>
              {STEPS.map((s, i) => (
                <li key={s.id}>
                  <button className={`${step === s.id ? "on" : ""} ${done[s.id] ? "done" : ""}`} aria-current={step === s.id ? "step" : undefined}
                    disabled={i > 0 && !proj.input} onClick={() => setStep(s.id)}>
                    <span className="n">{i + 1}</span><span><strong>{s.label}</strong><small>{s.hint}</small></span>
                  </button>
                </li>
              ))}
            </ol>
          </nav>
          <main className="main">
            <h1 className="song">{proj.name}</h1>
            {proj.performance && step !== "import" && step !== "export" && (
              <PianoRoll notes={proj.performance.notes} analysis={analysis} bpm={proj.performance.tempo.bpm}
                beatOffset={proj.performance.tempo.beat_offset || 0} flavouredNotes={flav?.notes} playhead={playhead} onChange={onNotes} />
            )}
            <div className="panel">
              {step === "import" && <ImportStep {...props} />}
              {step === "review" && <AnalysisStep {...props} analysis={analysis} savePerf={savePerf} />}
              {step === "voice" && <VoiceStep {...props} flav={flav} />}
              {step === "flavour" && <FlavourStep {...props} flav={flav} />}
              {step === "render" && <RenderStep {...props} job={job} onTime={setPlayhead} />}
              {step === "export" && <ExportStep {...props} />}
            </div>
            {stepIndex < STEPS.length - 1 && proj.performance && (
              <button className="next" onClick={() => setStep(STEPS[stepIndex + 1].id)}>Next: {STEPS[stepIndex + 1].label}</button>
            )}
          </main>
        </div>
      )}

      {(job || err) && (
        <footer className={`jobbar ${job?.status || ""}`} aria-live="polite">
          {err && <p className="error" role="alert">{err}</p>}
          {job && (
            <>
              <div className="jobline">
                <span>{job.kind}: {job.status === "running" || job.status === "queued" ? job.message || "working" : job.status}</span>
                <progress max={1} value={job.progress} />
                {(job.status === "running" || job.status === "queued") && <button className="danger" onClick={() => api.cancel(job.id)}>Cancel</button>}
                <button className="ghost" onClick={() => setShowLogs(!showLogs)}>{showLogs ? "Hide log" : "Show log"}</button>
                {job.status !== "running" && job.status !== "queued" && <button className="ghost" onClick={() => { setJob(null); setErr(null); }} aria-label="Dismiss">×</button>}
              </div>
              {showLogs && <pre className="logs">{job.logs.join("\n")}</pre>}
            </>
          )}
        </footer>
      )}
    </div>
  );
}
