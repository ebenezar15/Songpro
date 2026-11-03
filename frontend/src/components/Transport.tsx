import { useEffect, useRef, useState } from "react";

// Sample-synced A/B player: all buffers start from the same zero point;
// switching A/B changes gains only, so position never drifts.
type Props = { base: string; stems: string[]; version: number; onTime?: (t: number | null) => void; offset?: number };

export default function Transport({ base, stems, version, onTime, offset = 0 }: Props) {
  const ctx = useRef<AudioContext | null>(null);
  const bufs = useRef<Record<string, AudioBuffer>>({});
  const nodes = useRef<{ src: AudioBufferSourceNode; gain: GainNode; name: string }[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [side, setSide] = useState<"A" | "B">("B");
  const [mute, setMute] = useState<Record<string, boolean>>({});
  const [pos, setPos] = useState(0);
  const [dur, setDur] = useState(0);
  const startAt = useRef(0);
  const startPos = useRef(0);
  const raf = useRef(0);

  useEffect(() => {
    let dead = false;
    stop();
    setLoaded(false);
    setErr(null);
    (async () => {
      try {
        ctx.current = ctx.current || new AudioContext();
        const names = ["original", ...stems];
        const out: Record<string, AudioBuffer> = {};
        for (const n of names) {
          const r = await fetch(`${base}/${n}.wav?v=${version}`);
          if (!r.ok) throw new Error(`Could not load ${n}`);
          out[n] = await ctx.current.decodeAudioData(await r.arrayBuffer());
        }
        if (dead) return;
        bufs.current = out;
        setDur(Math.max(...Object.values(out).map((b) => b.duration)));
        setPos(0);
        setLoaded(true);
      } catch (e: any) {
        setErr(e.message);
      }
    })();
    return () => { dead = true; stop(); };
  }, [base, version, stems.join(",")]);

  const gainFor = (name: string, s = side, m = mute) =>
    name === "original" ? (s === "A" ? 1 : 0) : s === "B" && !m[name] ? 1 : 0;

  function play(from = pos) {
    const c = ctx.current!;
    c.resume();
    stop(false);
    const t = c.currentTime + 0.05;
    nodes.current = Object.entries(bufs.current).map(([name, b]) => {
      const src = c.createBufferSource();
      src.buffer = b;
      const gain = c.createGain();
      gain.gain.value = gainFor(name);
      src.connect(gain).connect(c.destination);
      if (from < b.duration) src.start(t, from);
      return { src, gain, name };
    });
    startAt.current = t;
    startPos.current = from;
    setPlaying(true);
    const tick = () => {
      const p = startPos.current + (c.currentTime - startAt.current);
      if (p >= dur) { stop(); setPos(0); onTime?.(null); return; }
      setPos(p);
      onTime?.(p + offset);
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
  }
  function stop(reset = true) {
    cancelAnimationFrame(raf.current);
    for (const n of nodes.current) { try { n.src.stop(); } catch { /* not started */ } }
    nodes.current = [];
    if (reset) setPlaying(false);
  }
  function switchSide(s: "A" | "B") {
    setSide(s);
    for (const n of nodes.current) n.gain.gain.setTargetAtTime(gainFor(n.name, s), ctx.current!.currentTime, 0.005);
  }
  function toggleMute(name: string) {
    const m = { ...mute, [name]: !mute[name] };
    setMute(m);
    for (const n of nodes.current) n.gain.gain.setTargetAtTime(gainFor(n.name, side, m), ctx.current!.currentTime, 0.005);
  }

  if (err) return <p className="error" role="alert">{err}</p>;
  if (!loaded) return <p className="muted">Loading audio…</p>;
  return (
    <div className="transport">
      <button className="primary round" onClick={() => (playing ? (stop(), onTime?.(null)) : play())} aria-label={playing ? "Stop" : "Play"}>
        {playing ? "■" : "▶"}
      </button>
      <div className="ab" role="radiogroup" aria-label="Compare">
        <button role="radio" aria-checked={side === "A"} className={side === "A" ? "on" : ""} onClick={() => switchSide("A")}>A · original</button>
        <button role="radio" aria-checked={side === "B"} className={side === "B" ? "on" : ""} onClick={() => switchSide("B")}>B · generated</button>
      </div>
      <input type="range" min={0} max={dur} step={0.01} value={pos} aria-label="Position"
        onChange={(e) => { const p = +e.target.value; setPos(p); if (playing) play(p); }} />
      <span className="time">{pos.toFixed(1)} / {dur.toFixed(1)} s</span>
      <div className="stems">
        {stems.map((s) => (
          <label key={s} className="check"><input type="checkbox" checked={!mute[s]} onChange={() => toggleMute(s)} /> {s}</label>
        ))}
      </div>
    </div>
  );
}
