// Records the microphone straight to 16-bit PCM WAV in the browser, so imports never need ffmpeg.
export type Recorder = { stop: () => Promise<Blob> };

export async function startWavRecording(): Promise<Recorder> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  const ctx = new AudioContext();
  const src = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  const chunks: Float32Array[] = [];
  proc.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  const mute = ctx.createGain();
  mute.gain.value = 0; // keep the graph running without echoing the mic to the speakers
  src.connect(proc).connect(mute).connect(ctx.destination);
  return {
    stop: async () => {
      src.disconnect(); proc.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      const sr = ctx.sampleRate;
      await ctx.close();
      const n = chunks.reduce((a, c) => a + c.length, 0);
      const buf = new ArrayBuffer(44 + n * 2);
      const v = new DataView(buf);
      const w = (o: number, s: string) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
      w(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); w(8, "WAVE"); w(12, "fmt ");
      v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
      v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
      w(36, "data"); v.setUint32(40, n * 2, true);
      let o = 44;
      for (const c of chunks) for (let i = 0; i < c.length; i++, o += 2) v.setInt16(o, Math.max(-1, Math.min(1, c[i])) * 0x7fff, true);
      return new Blob([buf], { type: "audio/wav" });
    },
  };
}
