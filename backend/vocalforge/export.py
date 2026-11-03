"""Logic Pro export: sample-aligned WAV stems (24-bit PCM / 32-bit float, optional BWF),
MIDI (notes + tempo map + lyrics), JSON session. No normalising, limiting or truncating:
clipping is prevented by one common gain applied to every stem (logged in the session)."""
from __future__ import annotations

import datetime as _dt
import io
import json
import re
import struct
import zipfile

import numpy as np
from scipy.signal import butter, sosfilt

from .audio_io import resample

HEADROOM_PEAK = 10 ** (-1.0 / 20)  # keep true sample peaks <= -1 dBFS
STEM_LABELS = {"lead": "LeadDry", "lead_processed": "LeadProcessed", "doubles": "Doubles",
               "harmonies": "Harmonies", "adlibs": "AdLibs", "instrumental": "Instrumental"}


# ------------------------------------------------------------------ WAV
def _bext_chunk(description: str, time_reference: int, origin="VocalForge"):
    now = _dt.datetime.now()
    b = bytearray(602)
    def put(off, size, s):
        e = s.encode("ascii", "replace")[:size]
        b[off:off + len(e)] = e
    put(0, 256, description)
    put(256, 32, origin)
    put(288, 32, "VocalForge")
    put(320, 10, now.strftime("%Y-%m-%d"))
    put(330, 8, now.strftime("%H:%M:%S"))
    struct.pack_into("<II", b, 338, time_reference & 0xFFFFFFFF, time_reference >> 32)
    struct.pack_into("<H", b, 346, 1)  # BWF version 1
    hist = b"A=PCM,F=VocalForge\r\n"
    data = bytes(b) + hist
    if len(data) % 2:
        data += b"\0"
    return b"bext" + struct.pack("<I", len(data)) + data


def wav_bytes(x: np.ndarray, sr: int, bit_depth: str = "24", bwf: bool = False, time_reference: int = 0,
              description: str = "") -> bytes:
    """x: float array [frames] or [frames, channels]."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n, ch = x.shape
    if bit_depth == "32f":
        fmt_tag, bits = 3, 32
        data = x.astype("<f4").tobytes()
    elif bit_depth == "24":
        fmt_tag, bits = 1, 24
        q = np.clip(np.round(x * 8388607.0), -8388608, 8388607).astype("<i4")
        data = q.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    elif bit_depth == "16":
        fmt_tag, bits = 1, 16
        data = np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2").tobytes()
    else:
        raise ValueError("bit_depth must be '24', '32f' or '16'")
    block = ch * bits // 8
    if fmt_tag == 3:
        fmt = struct.pack("<HHIIHHH", fmt_tag, ch, sr, sr * block, block, bits, 0)
    else:
        fmt = struct.pack("<HHIIHH", fmt_tag, ch, sr, sr * block, block, bits)
    chunks = b""
    if bwf:
        chunks += _bext_chunk(description, time_reference)
    chunks += b"fmt " + struct.pack("<I", len(fmt)) + fmt
    if fmt_tag == 3:
        chunks += b"fact" + struct.pack("<II", 4, n)
    pad = b"\0" if len(data) % 2 else b""
    chunks += b"data" + struct.pack("<I", len(data)) + data + pad
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def read_wav_header(buf: bytes) -> dict:
    assert buf[:4] == b"RIFF" and buf[8:12] == b"WAVE", "not a RIFF/WAVE file"
    pos = 12
    info = {"chunks": []}
    while pos + 8 <= len(buf):
        cid = buf[pos:pos + 4].decode("ascii", "replace")
        size = struct.unpack("<I", buf[pos + 4:pos + 8])[0]
        info["chunks"].append(cid)
        body = buf[pos + 8:pos + 8 + size]
        if cid == "fmt ":
            tag, ch, sr, _, block, bits = struct.unpack("<HHIIHH", body[:16])
            info.update(format_tag=tag, channels=ch, sample_rate=sr, block_align=block, bits=bits)
        elif cid == "data":
            info["frames"] = size // info["block_align"]
        elif cid == "bext":
            lo, hi = struct.unpack("<II", body[338:346])
            info["time_reference"] = lo | (hi << 32)
        pos += 8 + size + (size % 2)
    return info


# ------------------------------------------------------------------ MIDI
def _vlq(v: int) -> bytes:
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append(0x80 | (v & 0x7F))
        v >>= 7
    return bytes(reversed(out))


def _sec_to_ticks(t, tempo_changes, ppq):
    ticks = 0.0
    segs = sorted(tempo_changes, key=lambda c: c["time"])
    for i, c in enumerate(segs):
        st = c["time"]
        en = segs[i + 1]["time"] if i + 1 < len(segs) else float("inf")
        if t <= st:
            break
        span = min(t, en) - st
        ticks += span * c["bpm"] / 60.0 * ppq
    return int(round(ticks))


def midi_bytes(notes, tempo_changes, key=None, ppq=480, name="Lead", transpose=0):
    tempo_changes = tempo_changes or [{"time": 0.0, "bpm": 120.0}]
    if tempo_changes[0]["time"] > 0:
        tempo_changes = [{"time": 0.0, "bpm": tempo_changes[0]["bpm"]}] + tempo_changes
    # track 0: tempo map, time signature, key
    ev = []
    for c in tempo_changes:
        us = int(round(60_000_000 / c["bpm"]))
        ev.append((_sec_to_ticks(c["time"], tempo_changes, ppq), b"\xff\x51\x03" + us.to_bytes(3, "big")))
    ev.append((0, b"\xff\x58\x04\x04\x02\x18\x08"))
    if key:
        sharps = {"C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "C#": 7}
        rel_major = key["tonic"] if key.get("mode") == "major" else ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"][(key.get("tonic_pc", 0) + 3) % 12]
        sf_ = sharps.get(rel_major, 0)
        ev.append((0, b"\xff\x59\x02" + struct.pack("b", sf_) + (b"\x00" if key.get("mode") == "major" else b"\x01")))
    t0 = _track(ev)
    # track 1: notes + lyrics
    ev = [(0, b"\xff\x03" + _vlq(len(name.encode())) + name.encode())]
    for n in sorted(notes, key=lambda n: n["start"]):
        st = n["start"] + float((n.get("fx") or {}).get("timing_ms", 0)) / 1000
        on = _sec_to_ticks(max(0.0, st), tempo_changes, ppq)
        off = max(on + 1, _sec_to_ticks(st + n["dur"], tempo_changes, ppq))
        pitch = int(np.clip(round(n["midi"] + transpose), 0, 127))
        vel = int(np.clip(round(n.get("velocity", 0.8) * 127), 1, 127))
        lyr = (n.get("lyric") or "").encode("utf-8")
        if lyr and lyr != b"-":
            ev.append((on, b"\xff\x05" + _vlq(len(lyr)) + lyr))
        ev.append((on, bytes([0x90, pitch, vel])))
        ev.append((off, bytes([0x80, pitch, 0])))
    t1 = _track(ev)
    return b"MThd" + struct.pack(">IHHH", 6, 1, 2, ppq) + t0 + t1


def _track(events):
    events = sorted(events, key=lambda e: (e[0], 0 if e[1][0] == 0x80 else 1))
    out = b""
    last = 0
    for t, data in events:
        out += _vlq(t - last) + data
        last = t
    out += b"\x00\xff\x2f\x00"
    return b"MTrk" + struct.pack(">I", len(out)) + out


# --------------------------------------------------------------- processing
def process_lead(x, sr):
    """Optional 'processed' version: HPF 80 Hz, gentle compression, presence lift, short room."""
    y = sosfilt(butter(2, 80, "highpass", fs=sr, output="sos"), x)
    env = np.abs(y)
    a = np.exp(-1 / (0.01 * sr))
    # smooth envelope (vectorised one-pole via lfilter)
    from scipy.signal import lfilter
    env = lfilter([1 - a], [1, -a], env)
    thr = 10 ** (-18 / 20) * (np.max(env) + 1e-9)
    gain = np.where(env > thr, (thr / (env + 1e-12)) ** (1 - 1 / 3.0), 1.0)
    y = y * gain
    pres = sosfilt(butter(2, [2500, 6000], "bandpass", fs=sr, output="sos"), y)
    y = y + 0.25 * pres
    # tiny Schroeder room
    rev = np.zeros_like(y)
    for d, g in ((1557, 0.77), (1617, 0.8), (1491, 0.75), (1422, 0.78)):
        d = int(d * sr / 44100)
        b = np.zeros(d + 1); b[0] = 1
        aa = np.zeros(d + 1); aa[0] = 1; aa[d] = -g
        rev += lfilter(b, aa, y)
    return y + 0.08 * rev / 4


# -------------------------------------------------------------- packaging
def _slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "", s.title()) or "Song"


def stem_filename(song, stem, gender, age, bpm, key_short, sr, bit_depth):
    srs = f"{sr / 1000:g}k"
    bd = "32f" if bit_depth == "32f" else f"{bit_depth}b"
    age_s = {"child": "Child", "teen": "Teen", "young_adult": "YoungAdult", "adult": "Adult", "mature": "Mature", "senior": "Senior"}[age]
    return f"{_slug(song)}_{STEM_LABELS.get(stem, _slug(stem))}_{gender.title()}_{age_s}_{int(round(bpm))}bpm_{key_short}_{srs}{bd}.wav"


def build_export(stems: dict, native_sr: int, opts: dict, session: dict, notes, tempo_changes, key) -> tuple[bytes, dict]:
    """stems: name -> mono float32 at native_sr, all starting at song t=0.
    Returns (zip bytes, report)."""
    sr = int(opts.get("sample_rate", 48000))
    bit_depth = str(opts.get("bit_depth", "24"))
    channels = int(opts.get("channels", 1))
    bwf = bool(opts.get("bwf", False))
    include = opts.get("stems") or list(stems.keys())
    rendered = {}
    for name, x in stems.items():
        if name in include or name in ("lead", "instrumental"):
            rendered[name] = resample(np.asarray(x, np.float32), native_sr, sr)
    if opts.get("processed"):
        rendered["lead_processed"] = process_lead(rendered["lead"].astype(np.float64), sr).astype(np.float32)
    # sample alignment: pad all to identical length from the same zero point (never truncate)
    L = max(len(v) for v in rendered.values())
    for k in rendered:
        v = rendered[k]
        rendered[k] = np.pad(v, ((0, L - len(v)),) + ((0, 0),) * (v.ndim - 1))
    # one common gain so nothing clips (never boosts)
    peak = max(float(np.max(np.abs(v))) for v in rendered.values())
    gain = min(1.0, HEADROOM_PEAK / peak) if peak > 0 else 1.0
    song = opts.get("song_name", "Song")
    v = session.get("voice", {})
    bpm = session.get("tempo", {}).get("bpm", 120)
    key_short = (key or {}).get("short", "Cmaj")
    tref = int(opts.get("bwf_time_reference_s", 0) * sr)
    files = {}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, x in rendered.items():
            y = x.astype(np.float64) * gain
            if y.ndim == 2 and channels == 1:
                y = y.mean(axis=1)
            elif y.ndim == 2 and y.shape[1] == 1:
                y = y[:, 0]
            if channels == 2 and y.ndim == 1:
                y = np.stack([y, y], axis=1)
            fn = stem_filename(song, name, v.get("gender", "female"), v.get("age", "young_adult"), bpm, key_short, sr, bit_depth)
            data = wav_bytes(y, sr, bit_depth, bwf=bwf, time_reference=tref, description=f"VocalForge {name}")
            z.writestr(fn, data)
            files[name] = {"file": fn, "frames": L, "peak_dbfs": round(20 * np.log10(np.max(np.abs(y)) + 1e-12), 2)}
        if opts.get("midi", True):
            mid = midi_bytes(notes, tempo_changes, key, transpose=session.get("transpose", 0))
            z.writestr(f"{_slug(song)}_Notes_{int(round(bpm))}bpm.mid", mid)
        report = {"sample_rate": sr, "bit_depth": bit_depth, "channels": channels, "frames": L,
                  "duration_s": round(L / sr, 4), "gain_applied_db": round(20 * np.log10(gain), 2),
                  "normalised": False, "limited": False, "truncated": False, "bwf": bwf, "files": files}
        session = {**session, "export": report}
        z.writestr(f"{_slug(song)}_Session.json", json.dumps(session, indent=2, ensure_ascii=False, default=str))
        z.writestr("LOGIC_IMPORT_GUIDE.txt", LOGIC_GUIDE)
    return buf.getvalue(), report


LOGIC_GUIDE = """Importing VocalForge stems into Logic Pro
==========================================
1. Create a new project. Set File > Project Settings > Audio > Sample Rate to the
   rate in the file names (e.g. 48k) and set the tempo to the BPM in the file names.
2. Select all WAV files in Finder and drag them into the empty track area while
   holding nothing. Choose "Create new tracks" and "Place all files on bar 1 / at
   project start". Every stem starts at the same zero point, so they line up.
3. (Optional) Drag the .mid file in too: it contains the notes, lyrics and the
   tempo map. In the import dialog, answer "Import tempo" if you want Logic to follow
   tempo changes.
4. If you exported with BWF timestamps, you can instead select the regions and use
   Edit > Move > To Recorded Position.
5. Stems are dry by default. Files are not normalised or limited; a single common
   gain (see Session.json > export.gain_applied_db) was applied only if needed to
   keep peaks below -1 dBFS.
"""
