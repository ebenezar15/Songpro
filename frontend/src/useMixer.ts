import { useCallback, useEffect, useRef, useState } from "react";

// One shared Web Audio clock for every lane. Lanes are sample-aligned on the song timeline:
// a lane with offset 12.5 starts playing 12.5 s into the song (used by 8-bar previews).
export type MixLane = { id: string; url: string; offset: number; audible: boolean };

export function useMixer(lanes: MixLane[]) {
  const ctx = useRef<AudioContext | null>(null);
  const bufs = useRef<Map<string, AudioBuffer>>(new Map());
  const loadingUrls = useRef<Map<string, Promise<AudioBuffer>>>(new Map());
  const live = useRef<{ id: string; src: AudioBufferSourceNode; gain: GainNode }[]>([]);
  const [playing, setPlaying] = useState(false);
  const [pos, setPos] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const t0 = useRef(0);
  const p0 = useRef(0);
  const raf = useRef(0);
  const lanesRef = useRef(lanes);
  lanesRef.current = lanes;

  const load = useCallback(async (url: string) => {
    if (bufs.current.has(url)) return bufs.current.get(url)!;
    if (!loadingUrls.current.has(url)) {
      loadingUrls.current.set(url, (async () => {
        const r = await fetch(url);
        if (!r.ok) throw new Error(`Could not load audio (${r.status})`);
        const b = await ctx.current!.decodeAudioData(await r.arrayBuffer());
        bufs.current.set(url, b);
        return b;
      })());
    }
    return loadingUrls.current.get(url)!;
  }, []);

  const stop = useCallback(() => {
    cancelAnimationFrame(raf.current);
    for (const n of live.current) { try { n.src.stop(); } catch { /* not started */ } }
    live.current = [];
    setPlaying(false);
  }, []);

  const play = useCallback(async (from?: number) => {
    ctx.current = ctx.current || new AudioContext();
    const c = ctx.current;
    await c.resume();
    stop();
    const start = from ?? p0.current;
    setLoading(true);
    setError(null);
    try {
      // decode every lane so A/B switching never waits
      await Promise.all(lanesRef.current.map((l) => load(l.url)));
    } catch (e: any) {
      setError(e.message);
      setLoading(false);
      return;
    }
    setLoading(false);
    const when = c.currentTime + 0.06;
    live.current = lanesRef.current.map((l) => {
      const b = bufs.current.get(l.url)!;
      const src = c.createBufferSource();
      src.buffer = b;
      const gain = c.createGain();
      gain.gain.value = l.audible ? 1 : 0;
      src.connect(gain).connect(c.destination);
      const rel = start - l.offset;
      if (rel < b.duration) {
        if (rel >= 0) src.start(when, rel);
        else src.start(when - rel, 0);
      }
      return { id: l.id, src, gain };
    });
    t0.current = when;
    p0.current = start;
    setPlaying(true);
    const end = Math.max(...lanesRef.current.map((l) => l.offset + (bufs.current.get(l.url)?.duration ?? 0)));
    const tick = () => {
      const p = p0.current + Math.max(0, c.currentTime - t0.current);
      if (p >= end) { stop(); p0.current = 0; setPos(0); return; }
      setPos(p);
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
  }, [load, stop]);

  const pause = useCallback(() => {
    if (!ctx.current) return;
    p0.current = p0.current + Math.max(0, ctx.current.currentTime - t0.current);
    setPos(p0.current);
    stop();
  }, [stop]);

  const seek = useCallback((t: number) => {
    p0.current = Math.max(0, t);
    setPos(p0.current);
    if (live.current.length) play(p0.current);
  }, [play]);

  // mute / solo / compare changes apply instantly without restarting
  useEffect(() => {
    if (!ctx.current) return;
    for (const n of live.current) {
      const l = lanes.find((x) => x.id === n.id);
      n.gain.gain.setTargetAtTime(l && l.audible ? 1 : 0, ctx.current.currentTime, 0.008);
    }
  }, [lanes.map((l) => `${l.id}:${l.audible}`).join(",")]);

  // a lane whose audio changed (new render) must be re-fetched next time
  useEffect(() => {
    const urls = new Set(lanes.map((l) => l.url));
    for (const k of [...bufs.current.keys()]) if (!urls.has(k)) { bufs.current.delete(k); loadingUrls.current.delete(k); }
    if (live.current.length) stop();
  }, [lanes.map((l) => l.url).join(",")]);

  useEffect(() => () => stop(), [stop]);
  return { play, pause, seek, stop, playing, pos, loading, error };
}
