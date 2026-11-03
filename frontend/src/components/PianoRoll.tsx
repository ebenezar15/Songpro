import { useEffect, useMemo, useRef, useState } from "react";
import { Note, midiName } from "../api";

type Props = {
  notes: Note[];
  analysis: any | null;
  bpm: number;
  beatOffset: number;
  changes?: any[];
  flavouredNotes?: Note[] | null;
  playhead?: number | null;
  onChange: (notes: Note[]) => void;
  pxs: number;
  scrollX: number;
  onView: (pxs: number, scrollX: number) => void;
  onSeek?: (t: number) => void;
};

export const KEY_W = 46;
const LANE_H = 44; // loudness lane


function css(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export default function PianoRoll({ notes, analysis, bpm, beatOffset, changes, flavouredNotes, playhead, onChange, pxs, scrollX, onView, onSeek }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const cv = useRef<HTMLCanvasElement>(null);
  const setScrollX = (v: number) => onView(pxs, v);
  const setPxs = (f: (p: number) => number) => onView(f(pxs), scrollX);
  const [sel, setSel] = useState<string | null>(null);
  const [snap, setSnap] = useState(true);
  const [showFlavour, setShowFlavour] = useState(true);
  const [edit, setEdit] = useState<{ id: string; x: number; y: number; text: string } | null>(null);
  const [width, setWidth] = useState(900);
  const drag = useRef<any>(null);

  const duration = analysis?.duration ?? Math.max(4, ...notes.map((n) => n.start + n.dur)) + 1;
  const [lo, hi] = useMemo(() => {
    const ms = notes.map((n) => n.midi);
    if (!ms.length) return [55, 76];
    return [Math.floor(Math.min(...ms)) - 4, Math.ceil(Math.max(...ms)) + 4];
  }, [notes.length === 0, Math.min(...notes.map((n) => n.midi), 999), Math.max(...notes.map((n) => n.midi), -1)]);
  const rows = hi - lo + 1;
  const ROW = rows > 34 ? 8 : rows > 26 ? 10 : 13;
  const height = rows * ROW + LANE_H + 22;

  useEffect(() => {
    const ro = new ResizeObserver((e) => setWidth(Math.max(320, e[0].contentRect.width)));
    if (wrap.current) ro.observe(wrap.current);
    return () => ro.disconnect();
  }, []);

  const tx = (t: number) => KEY_W + (t - scrollX) * pxs;
  const ty = (m: number) => 22 + (hi - m) * ROW;
  const xt = (x: number) => (x - KEY_W) / pxs + scrollX;
  const ym = (y: number) => hi - (y - 22) / ROW + 0.5;
  const beat = 60 / (bpm || 120);
  const grid = beat / 4;

  useEffect(() => {
    const c = cv.current!;
    const dpr = window.devicePixelRatio || 1;
    c.width = width * dpr;
    c.height = height * dpr;
    c.style.width = width + "px";
    c.style.height = height + "px";
    const g = c.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const C = {
      bg: css("--roll-bg"), row: css("--roll-row"), black: css("--roll-black"), line: css("--roll-line"),
      bar: css("--roll-bar"), note: css("--lead"), noteEdit: css("--ivory"), f0: css("--pitch"),
      warn: css("--warn"), text: css("--ivory"), muted: css("--dim"), flav: css("--harmony"),
    };
    g.fillStyle = C.bg;
    g.fillRect(0, 0, width, height);
    // rows
    for (let m = lo; m <= hi; m++) {
      const y = ty(m) - ROW / 2;
      const black = [1, 3, 6, 8, 10].includes(((m % 12) + 12) % 12);
      g.fillStyle = black ? C.black : C.row;
      g.fillRect(KEY_W, y, width, ROW);
    }
    // low-confidence regions
    for (const r of analysis?.low_confidence ?? []) {
      g.fillStyle = C.warn + "33";
      g.fillRect(tx(r.start), 22, (r.end - r.start) * pxs, rows * ROW);
    }
    // phrases + breaths ribbon
    g.font = "11px 'Anek Tamil Variable'";
    for (const p of analysis?.phrases ?? []) {
      g.fillStyle = C.line;
      g.fillRect(tx(p.start), 4, (p.end - p.start) * pxs, 12);
      if ((p.end - p.start) * pxs > 70) {
        g.fillStyle = C.muted;
        g.fillText(p.id.replace("p", "phrase "), tx(p.start) + 4, 14);
      }
    }
    for (const b of analysis?.breaths ?? []) {
      g.fillStyle = C.f0;
      g.beginPath();
      g.arc(tx((b.start + b.end) / 2), 10, 3, 0, Math.PI * 2);
      g.fill();
    }
    // beat grid
    const t0 = Math.max(0, scrollX - 1), t1 = scrollX + width / pxs;
    let k = Math.floor((t0 - beatOffset) / beat);
    const beatPx = beat * pxs;
    const labelEvery = Math.max(1, Math.ceil(28 / (beatPx * 4)));
    for (let t = beatOffset + k * beat; t < t1; t += beat, k++) {
      if (t < 0) continue;
      const bar = ((k % 4) + 4) % 4 === 0;
      if (!bar && beatPx < 7) continue;          // too dense to read: bars only
      if (bar && beatPx * 4 < 5) continue;
      g.strokeStyle = bar ? C.bar : C.line;
      g.lineWidth = bar ? 1.2 : 0.6;
      g.beginPath();
      g.moveTo(tx(t), 22);
      g.lineTo(tx(t), 22 + rows * ROW);
      g.stroke();
      const barNo = Math.round((t - beatOffset) / beat / 4);
      if (bar && barNo % labelEvery === 0) {
        g.fillStyle = C.muted;
        g.fillText(String(barNo + 1), tx(t) + 3, 22 + rows * ROW + 12);
      }
    }
    // flavoured (ghost) notes
    if (showFlavour && flavouredNotes) {
      g.strokeStyle = C.flav;
      g.setLineDash([3, 3]);
      for (const n of flavouredNotes) {
        if (!n.generated) continue;
        g.strokeRect(tx(n.start), ty(n.midi) - ROW / 2 + 1, n.dur * pxs, ROW - 2);
      }
      g.setLineDash([]);
    }
    // notes
    g.font = "600 12px 'Anek Tamil Variable'";
    let lyricEnd = -1e9;
    for (const n of notes) {
      const x = tx(n.start), y = ty(n.midi) - ROW / 2 + 1, w = Math.max(3, n.dur * pxs);
      if (x + w < KEY_W || x > width) continue;
      g.globalAlpha = n.continuation ? 0.55 : 1;
      g.fillStyle = n.edited ? C.noteEdit : C.note;
      g.fillRect(x, y, w, ROW - 2);
      if ((n.confidence ?? 1) < 0.7) {
        g.strokeStyle = C.warn;
        g.lineWidth = 2;
        g.strokeRect(x, y, w, ROW - 2);
      }
      if (n.id === sel) {
        g.strokeStyle = C.text;
        g.lineWidth = 2;
        g.strokeRect(x - 1, y - 1, w + 2, ROW);
      }
      g.globalAlpha = 1;
      if (n.lyric && n.lyric !== "-" && x > lyricEnd) {  // skip labels that would collide when zoomed out
        const tw = g.measureText(n.lyric).width;
        g.fillStyle = C.text;
        g.fillText(n.lyric, x + 2, y - 3);
        lyricEnd = x + tw + 6;
      }
      if (n.vibrato) {
        g.strokeStyle = C.bg;
        g.beginPath();
        for (let i = 0; i < w - 4; i += 2) g.lineTo(x + 2 + i, y + ROW / 2 - 1 + Math.sin(i / 2) * 2);
        g.stroke();
      }
    }
    // F0 contour
    if (analysis?.f0_times) {
      g.strokeStyle = C.f0;
      g.lineWidth = 1.6;
      g.globalAlpha = 0.9;
      g.beginPath();
      let pen = false;
      let lastM = 0;
      const ts = analysis.f0_times, fs = analysis.f0_hz;
      for (let i = 0; i < ts.length; i++) {
        const f = fs[i];
        if (ts[i] < t0 || ts[i] > t1) { pen = false; continue; }
        if (f <= 0 || (analysis.f0_conf && analysis.f0_conf[i] < 0.6)) { pen = false; continue; }
        const m = 69 + 12 * Math.log2(f / 440);
        if (m < lo - 1 || m > hi + 1) { pen = false; continue; }
        const x = tx(ts[i]), y = ty(m);
        if (pen && Math.abs(m - lastM) > 1.5) pen = false;
        if (!pen) { g.moveTo(x, y); pen = true; } else g.lineTo(x, y);
        lastM = m;
      }
      g.stroke();
      g.globalAlpha = 1;
    }
    // loudness lane
    const ly = 22 + rows * ROW + 16;
    g.fillStyle = C.black;
    g.fillRect(KEY_W, ly, width, LANE_H - 16);
    if (analysis?.loudness_db) {
      const ts = analysis.f0_times, db = analysis.loudness_db;
      const top = Math.max(...db), floor = top - 50;
      g.strokeStyle = C.note;
      g.beginPath();
      for (let i = 0; i < ts.length; i += 2) {
        if (ts[i] < t0 || ts[i] > t1) continue;
        const v = Math.max(0, (db[i] - floor) / 50);
        g.lineTo(tx(ts[i]), ly + (LANE_H - 16) * (1 - v));
      }
      g.stroke();
    }
    // keys
    for (let m = lo; m <= hi; m++) {
      const y = ty(m) - ROW / 2;
      const black = [1, 3, 6, 8, 10].includes(((m % 12) + 12) % 12);
      g.fillStyle = black ? C.black : C.row;
      g.fillRect(0, y, KEY_W, ROW);
      if (((m % 12) + 12) % 12 === 0) {
        g.fillStyle = C.muted;
        g.font = "11px 'Anek Tamil Variable'";
        g.fillText(midiName(m), 6, y + ROW - 2);
      }
    }
    if (playhead != null) {
      g.strokeStyle = C.text;
      g.lineWidth = 1.5;
      g.beginPath();
      g.moveTo(tx(playhead), 0);
      g.lineTo(tx(playhead), height);
      g.stroke();
    }
  }, [notes, analysis, width, height, pxs, scrollX, sel, bpm, beatOffset, flavouredNotes, showFlavour, playhead, lo, hi]);

  const hit = (x: number, y: number) => {
    for (let i = notes.length - 1; i >= 0; i--) {
      const n = notes[i];
      const nx = tx(n.start), ny = ty(n.midi) - ROW / 2, w = Math.max(3, n.dur * pxs);
      if (x >= nx && x <= nx + w && y >= ny && y <= ny + ROW) return { n, edge: x > nx + w - 6 };
    }
    return null;
  };
  const q = (t: number, alt: boolean) => (snap && !alt ? beatOffset + Math.round((t - beatOffset) / grid) * grid : t);

  function down(e: React.PointerEvent) {
    const r = cv.current!.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const h = hit(x, y);
    if (h) {
      setSel(h.n.id);
      drag.current = { id: h.n.id, mode: h.edge ? "resize" : "move", x0: x, y0: y, orig: { ...h.n } };
      (e.target as Element).setPointerCapture(e.pointerId);
    } else {
      setSel(null);
      drag.current = { mode: "pan", x0: x, s0: scrollX };
    }
  }
  function move(e: React.PointerEvent) {
    const d = drag.current;
    if (!d) return;
    const r = cv.current!.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    if (d.mode === "pan") {
      if (Math.abs(x - d.x0) > 3) d.moved = true;
      if (d.moved) setScrollX(Math.max(0, d.s0 - (x - d.x0) / pxs));
      return;
    }
    const dt = (x - d.x0) / pxs;
    const dm = Math.round((d.y0 - y) / ROW);
    const next = notes.map((n) => {
      if (n.id !== d.id) return n;
      if (d.mode === "move") {
        const st = Math.max(0, q(d.orig.start + dt, e.altKey));
        return { ...n, start: +st.toFixed(4), midi: Math.round(d.orig.midi) + dm, edited: true };
      }
      const end = q(d.orig.start + d.orig.dur + dt, e.altKey);
      return { ...n, dur: +Math.max(0.05, end - n.start).toFixed(4), edited: true };
    });
    d.moved = true;
    onChange(next);
  }
  function up() {
    const d = drag.current;
    if (d && d.mode === "pan" && !d.moved && onSeek) onSeek(Math.max(0, xt(d.x0)));
    drag.current = null;
  }
  function dbl(e: React.MouseEvent) {
    const r = cv.current!.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const h = hit(x, y);
    if (h) {
      setEdit({ id: h.n.id, x: tx(h.n.start), y: ty(h.n.midi) - 30, text: h.n.lyric ?? "" });
      return;
    }
    const st = q(xt(x), e.altKey);
    const n: Note = { id: `u${Date.now().toString(36)}`, start: +st.toFixed(4), dur: +(grid * 2).toFixed(4), midi: Math.round(ym(y)), velocity: 0.8, edited: true, confidence: 1, lyric: "" };
    onChange([...notes, n].sort((a, b) => a.start - b.start));
    setSel(n.id);
  }
  function key(e: React.KeyboardEvent) {
    if (!sel || edit) return;
    const n = notes.find((n) => n.id === sel);
    if (!n) return;
    if (e.key === "Delete" || e.key === "Backspace") {
      onChange(notes.filter((n) => n.id !== sel));
      setSel(null);
    } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      const dm = (e.key === "ArrowUp" ? 1 : -1) * (e.shiftKey ? 12 : 1);
      onChange(notes.map((m) => (m.id === sel ? { ...m, midi: Math.round(m.midi) + dm, edited: true } : m)));
    } else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      const dt = (e.key === "ArrowRight" ? 1 : -1) * grid;
      onChange(notes.map((m) => (m.id === sel ? { ...m, start: +Math.max(0, m.start + dt).toFixed(4), edited: true } : m)));
    } else return;
    e.preventDefault();
  }
  function wheel(e: React.WheelEvent) {
    if (e.ctrlKey || e.metaKey) setPxs((p) => Math.min(600, Math.max(20, p * (e.deltaY < 0 ? 1.15 : 0.87))));
    else if (Math.abs(e.deltaX) > Math.abs(e.deltaY) || e.shiftKey) setScrollX(Math.max(0, Math.min(duration, scrollX + (e.deltaX || e.deltaY) / pxs)));
  }

  const selNote = notes.find((n) => n.id === sel);
  return (
    <section className="roll" aria-label="Piano roll">
      <div className="roll-tools">
        <label className="check"><input type="checkbox" checked={snap} onChange={(e) => setSnap(e.target.checked)} /> Snap to 1/16</label>
        <label className="check"><input type="checkbox" checked={showFlavour} onChange={(e) => setShowFlavour(e.target.checked)} /> Show flavour notes</label>
        <span className="zoom">
          <button className="ghost" onClick={() => setPxs((p) => Math.max(20, p / 1.4))} aria-label="Zoom out">−</button>
          <button className="ghost" onClick={() => setPxs((p) => Math.min(600, p * 1.4))} aria-label="Zoom in">+</button>
        </span>
        <span className="roll-sel">
          {selNote
            ? `${midiName(selNote.midi)} · ${selNote.start.toFixed(2)}s · ${(selNote.dur * 1000).toFixed(0)} ms${selNote.ipa ? ` · /${selNote.ipa}/` : ""}${selNote.edited ? " · edited" : ""}`
            : "Drag a note to move it, drag its right edge to resize, double-click to add a note or type its lyric. Alt turns snap off."}
        </span>
      </div>
      <div className="roll-canvas" ref={wrap} tabIndex={0} onKeyDown={key} onWheel={wheel}>
        <canvas ref={cv} onPointerDown={down} onPointerMove={move} onPointerUp={up} onDoubleClick={dbl} />
        {edit && (
          <input
            className="lyric-edit"
            style={{ left: edit.x, top: Math.max(0, edit.y) }}
            autoFocus
            value={edit.text}
            onChange={(e) => setEdit({ ...edit, text: e.target.value })}
            onBlur={() => setEdit(null)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                onChange(notes.map((n) => (n.id === edit.id ? { ...n, lyric: edit.text, edited: true, lyric_dirty: true, continuation: edit.text === "-" } as Note : n)));
                setEdit(null);
              } else if (e.key === "Escape") setEdit(null);
            }}
            aria-label="Lyric for note"
          />
        )}
      </div>
    </section>
  );
}
