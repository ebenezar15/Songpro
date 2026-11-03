import { ReactNode, useEffect, useRef, useState } from "react";
import { fmtTime, peaksOf } from "../api";
import { KEY_W } from "./PianoRoll";

export type Lane = {
  id: string; label: string; color: string; src: string; offset: number; version: number;
  note?: string; download?: { href: string; label: string } | null;
};

type View = { pxs: number; scrollX: number };
type Props = {
  pid: string; lanes: Lane[]; view: View; onView: (v: View) => void; duration: number; bpm: number; beatOffset: number;
  playhead: number; onSeek: (t: number) => void; mute: Record<string, boolean>; solo: string | null;
  onMute: (id: string) => void; onSolo: (id: string) => void; roll?: ReactNode; rollLabel?: ReactNode;
};

function useWidth(ref: React.RefObject<HTMLElement | null>) {
  const [w, setW] = useState(800);
  useEffect(() => {
    const ro = new ResizeObserver((e) => setW(Math.max(240, e[0].contentRect.width)));
    if (ref.current) ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return w;
}

const cssVar = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function Ruler({ view, width, duration, bpm, beatOffset, playhead, onSeek }: { view: View; width: number; duration: number; bpm: number; beatOffset: number; playhead: number; onSeek: (t: number) => void }) {
  const cv = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = cv.current!, dpr = window.devicePixelRatio || 1, H = 30;
    c.width = width * dpr; c.height = H * dpr; c.style.width = width + "px"; c.style.height = H + "px";
    const g = c.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, width, H);
    const bar = (4 * 60) / (bpm || 120);
    const pxBar = bar * view.pxs;
    const every = pxBar < 24 ? Math.ceil(24 / pxBar / 4) * 4 : 1;
    g.font = "600 11px 'Anek Tamil Variable'";
    for (let k = Math.floor((view.scrollX - beatOffset) / bar); ; k++) {
      const t = beatOffset + k * bar;
      const x = KEY_W + (t - view.scrollX) * view.pxs;
      if (x > width) break;
      if (k < 0 || k % every) continue;
      g.fillStyle = cssVar("--rule-strong");
      g.fillRect(x, 16, 1, 14);
      g.fillStyle = cssVar("--dim");
      g.fillText(String(k + 1), x + 3, 26);
    }
    const tStep = view.pxs > 60 ? 5 : view.pxs > 15 ? 15 : 30;
    for (let t = Math.ceil(view.scrollX / tStep) * tStep; t < duration + tStep; t += tStep) {
      const x = KEY_W + (t - view.scrollX) * view.pxs;
      if (x > width) break;
      g.fillStyle = cssVar("--dim");
      g.fillText(fmtTime(t).replace(/\.\d$/, ""), x + 3, 11);
      g.fillRect(x, 0, 1, 5);
    }
    const px = KEY_W + (playhead - view.scrollX) * view.pxs;
    g.fillStyle = cssVar("--ivory");
    g.beginPath(); g.moveTo(px - 5, 0); g.lineTo(px + 5, 0); g.lineTo(px, 8); g.fill();
  }, [view, width, duration, bpm, beatOffset, playhead]);
  return <canvas ref={cv} className="ruler" onClick={(e) => {
    const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
    onSeek(Math.max(0, (e.clientX - r.left - KEY_W) / view.pxs + view.scrollX));
  }} aria-label="Timeline ruler, click to move the playhead" />;
}

function Wave({ pid, lane, view, width, playhead, onSeek, dim }: { pid: string; lane: Lane; view: View; width: number; playhead: number; onSeek: (t: number) => void; dim: boolean }) {
  const cv = useRef<HTMLCanvasElement>(null);
  const [pk, setPk] = useState<{ duration: number; min: number[]; max: number[] } | null>(null);
  const [err, setErr] = useState(false);
  useEffect(() => {
    setPk(null); setErr(false);
    peaksOf(pid, lane.src, 6000, lane.version).then(setPk).catch(() => setErr(true));
  }, [pid, lane.src, lane.version]);
  useEffect(() => {
    const c = cv.current!, dpr = window.devicePixelRatio || 1, H = 58;
    c.width = width * dpr; c.height = H * dpr; c.style.width = width + "px"; c.style.height = H + "px";
    const g = c.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, width, H);
    if (!pk) return;
    const n = pk.max.length, spp = pk.duration / n;
    let peak = 1e-6;
    for (let i = 0; i < n; i++) peak = Math.max(peak, pk.max[i], -pk.min[i]);
    g.fillStyle = cssVar(lane.color);
    g.globalAlpha = dim ? 0.3 : 0.9;
    const mid = H / 2;
    for (let x = KEY_W; x < width; x++) {
      const t0 = (x - KEY_W) / view.pxs + view.scrollX - lane.offset;
      const t1 = t0 + 1 / view.pxs;
      if (t1 < 0 || t0 > pk.duration) continue;
      const i0 = Math.max(0, Math.floor(t0 / spp)), i1 = Math.min(n, Math.max(i0 + 1, Math.ceil(t1 / spp)));
      let lo = 0, hi = 0;
      for (let i = i0; i < i1; i++) { if (pk.min[i] < lo) lo = pk.min[i]; if (pk.max[i] > hi) hi = pk.max[i]; }
      // scaled to the lane's own peak with gentle compression, so quiet and loud parts both read
      const a = Math.min(mid, Math.pow(hi / peak, 0.75) * mid * 0.95), b = Math.min(mid, Math.pow(-lo / peak, 0.75) * mid * 0.95);
      g.fillRect(x, mid - a, 1, Math.max(1, a + b));
    }
    g.globalAlpha = 1;
    const px = KEY_W + (playhead - view.scrollX) * view.pxs;
    g.fillStyle = cssVar("--ivory");
    g.fillRect(px, 0, 1.5, H);
  }, [pk, view, width, playhead, dim, lane.offset, lane.color]);
  return (
    <div className="wave">
      <canvas ref={cv} onClick={(e) => {
        const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
        onSeek(Math.max(0, (e.clientX - r.left - KEY_W) / view.pxs + view.scrollX));
      }} />
      {err && <span className="wave-msg">Waveform unavailable</span>}
      {!pk && !err && <span className="wave-msg">Drawing waveform…</span>}
    </div>
  );
}

export default function Timeline(p: Props) {
  const area = useRef<HTMLDivElement>(null);
  const width = useWidth(area);
  const { view, onView } = p;
  const visible = (width - KEY_W) / view.pxs;
  function wheel(e: React.WheelEvent) {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      onView({ ...view, pxs: Math.min(600, Math.max(2, view.pxs * (e.deltaY < 0 ? 1.15 : 0.87))) });
    } else if (Math.abs(e.deltaX) > Math.abs(e.deltaY) || e.shiftKey) {
      onView({ ...view, scrollX: Math.max(0, Math.min(p.duration, view.scrollX + (e.deltaX || e.deltaY) / view.pxs)) });
    }
  }
  return (
    <section className="timeline" aria-label="Timeline">
      <div className="tl-row tl-head">
        <div className="lane-head zoom-head">
          <button className="icon" onClick={() => onView({ ...view, pxs: Math.max(2, view.pxs / 1.5) })} aria-label="Zoom out">−</button>
          <button className="icon" onClick={() => onView({ ...view, pxs: Math.min(600, view.pxs * 1.5) })} aria-label="Zoom in">+</button>
          <button className="text-btn" onClick={() => onView({ pxs: Math.max(2, (width - KEY_W - 8) / Math.max(1, p.duration)), scrollX: 0 })}>Fit song</button>
        </div>
        <div className="lane-body" ref={area}>
          <Ruler view={view} width={width} duration={p.duration} bpm={p.bpm} beatOffset={p.beatOffset} playhead={p.playhead} onSeek={p.onSeek} />
        </div>
      </div>
      <div className="tl-lanes" onWheel={wheel}>
        {p.lanes.map((l) => {
          const audible = p.solo ? p.solo === l.id : !p.mute[l.id];
          return (
            <div key={l.id} className="tl-row" style={{ ["--lane" as any]: `var(${l.color})` }}>
              <div className="lane-head">
                <span className="swatch" aria-hidden />
                <div className="lane-title"><strong>{l.label}</strong>{l.note && <small>{l.note}</small>}</div>
                <div className="lane-btns">
                  <button className={`ms ${p.mute[l.id] ? "on" : ""}`} aria-pressed={!!p.mute[l.id]} onClick={() => p.onMute(l.id)} title="Mute">M</button>
                  <button className={`ms solo ${p.solo === l.id ? "on" : ""}`} aria-pressed={p.solo === l.id} onClick={() => p.onSolo(l.id)} title="Solo">S</button>
                  {l.download && <a className="ms dl" href={l.download.href} download title={l.download.label} aria-label={l.download.label}>↓</a>}
                </div>
              </div>
              <div className="lane-body">
                <Wave pid={p.pid} lane={l} view={view} width={width} playhead={p.playhead} onSeek={p.onSeek} dim={!audible} />
              </div>
            </div>
          );
        })}
        {p.roll && (
          <div className="tl-row roll-row" style={{ ["--lane" as any]: "var(--lead)" }}>
            <div className="lane-head"><span className="swatch" aria-hidden /><div className="lane-title">{p.rollLabel}</div></div>
            <div className="lane-body">{p.roll}</div>
          </div>
        )}
      </div>
      <div className="tl-row">
        <div className="lane-head" />
        <div className="lane-body">
          <input className="tl-scroll" type="range" min={0} max={Math.max(0, p.duration - visible * 0.5)} step={0.01} value={view.scrollX}
            onChange={(e) => onView({ ...view, scrollX: +e.target.value })} aria-label="Scroll the timeline" />
        </div>
      </div>
    </section>
  );
}
