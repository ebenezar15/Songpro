// Thin client for the local VocalForge API (127.0.0.1 only).
export type Note = {
  id: string; start: number; dur: number; midi: number; velocity?: number; confidence?: number;
  lyric?: string; ipa?: string; phonemes?: { onset: string[]; vowel: string; coda: string[] } | null;
  continuation?: boolean; edited?: boolean; vibrato?: { rate_hz: number; depth_cents: number } | null;
  sustained?: boolean; slide_in?: boolean; generated?: string; lyric_dirty?: boolean;
};
export type Job = { id: string; kind: string; status: string; progress: number; message: string; logs: string[]; result: any; error: string | null };

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const b = await r.json(); msg = b.detail || msg; } catch { /* keep status */ }
    throw new Error(msg);
  }
  return r.json();
}
const H = { "Content-Type": "application/json" };
export const api = {
  meta: () => fetch("/api/meta").then(j<any>),
  system: () => fetch("/api/system").then(j<any>),
  projects: () => fetch("/api/projects").then(j<any[]>),
  create: (name: string) => fetch("/api/projects", { method: "POST", headers: H, body: JSON.stringify({ name }) }).then(j<any>),
  project: (id: string) => fetch(`/api/projects/${id}`).then(j<any>),
  update: (id: string, body: any) => fetch(`/api/projects/${id}`, { method: "PUT", headers: H, body: JSON.stringify(body) }).then(j<any>),
  performance: (id: string, body: any) => fetch(`/api/projects/${id}/performance`, { method: "PUT", headers: H, body: JSON.stringify(body) }).then(j<any>),
  resetPerformance: (id: string) => fetch(`/api/projects/${id}/performance/reset`, { method: "POST" }).then(j<any>),
  upload: (id: string, file: Blob, name: string, role: "input" | "reference", consent = false) => {
    const fd = new FormData();
    fd.append("file", file, name); fd.append("role", role); fd.append("consent", String(consent));
    return fetch(`/api/projects/${id}/audio`, { method: "POST", body: fd }).then(j<any>);
  },
  analyze: (id: string, force = false, useSeparated = false) =>
    fetch(`/api/projects/${id}/analyze?force=${force}&use_separated=${useSeparated}`, { method: "POST" }).then(j<Job>),
  analysis: (id: string) => fetch(`/api/projects/${id}/analysis`).then(j<any>),
  separate: (id: string) => fetch(`/api/projects/${id}/separate`, { method: "POST" }).then(j<Job>),
  parseLyrics: (text: string, language: string, dialect: string, glides: boolean) =>
    fetch("/api/lyrics/parse", { method: "POST", headers: H, body: JSON.stringify({ text, language, dialect, glides }) }).then(j<any>),
  flavour: (id: string) => fetch(`/api/projects/${id}/flavour`).then(j<any>),
  render: (id: string, mode: "preview" | "full", start_bar = 0, bars = 8) =>
    fetch(`/api/projects/${id}/render`, { method: "POST", headers: H, body: JSON.stringify({ mode, start_bar, bars }) }).then(j<Job>),
  renderMeta: (id: string, mode: string) => fetch(`/api/projects/${id}/renders/${mode}`).then(j<any>),
  export: (id: string, opts: any) => fetch(`/api/projects/${id}/export`, { method: "POST", headers: H, body: JSON.stringify(opts) }).then(j<Job>),
  job: (id: string) => fetch(`/api/jobs/${id}`).then(j<Job>),
  cancel: (id: string) => fetch(`/api/jobs/${id}/cancel`, { method: "POST" }).then(j<Job>),
  consent: (model: string, consent: boolean) =>
    fetch(`/api/models/${model}/consent`, { method: "POST", headers: H, body: JSON.stringify({ consent }) }).then(j<any>),
};

export async function waitJob(job: Job, onTick: (j: Job) => void): Promise<Job> {
  let cur = job;
  onTick(cur);
  while (cur.status === "queued" || cur.status === "running") {
    await new Promise((r) => setTimeout(r, 250));
    cur = await api.job(job.id);
    onTick(cur);
  }
  return cur;
}

export const NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
export const midiName = (m: number) => `${NOTE_NAMES[((Math.round(m) % 12) + 12) % 12]}${Math.floor(Math.round(m) / 12) - 1}`;
