"""Lyrics -> syllables with phonemes, and syllable -> note alignment."""
from __future__ import annotations

import re

from .english import english_word_to_syllables
from .tamil import is_tamil_script, roman_word_to_syllables, tamil_word_to_syllables

LANGUAGES = {
    "ta": "Tamil (தமிழ்)",
    "ta-Latn": "Tamil, romanised (Tanglish)",
    "en": "English",
}
HOLD_TOKENS = {"-", "_", "~"}


def parse_lyrics(text: str, language: str = "ta", dialect: str = "standard", glides: bool = True):
    """Return a flat list of syllables. '-' tokens extend the previous syllable (melisma).
    Each line becomes a phrase hint."""
    out = []
    for line_no, line in enumerate(text.splitlines()):
        for tok in re.findall(r"[^\s]+", line):
            if tok in HOLD_TOKENS:
                out.append({"text": "-", "hold": True, "line": line_no})
                continue
            # explicit syllable split with '.' or '·' (e.g. "kaa.dhal")
            word = re.sub(r"[,!?;:\"“”‘’()\[\]]", "", tok)
            if not word:
                continue
            if is_tamil_script(word):
                sylls = tamil_word_to_syllables(word, dialect=dialect, glides=glides)
                lang = "ta"
            elif language in ("ta", "ta-Latn"):
                parts = re.split(r"[.·]", word)
                sylls = [s for p in parts for s in roman_word_to_syllables(p)]
                lang = "ta-Latn"
            else:
                sylls = english_word_to_syllables(word.lower())
                lang = "en"
            for k, s in enumerate(sylls):
                s.update({"word": word, "word_start": k == 0, "line": line_no, "lang": lang, "hold": False})
                s["ipa"] = "".join(s["onset"]) + s["vowel"] + "".join(s["coda"])
                out.append(s)
    return out


def align_to_notes(notes: list[dict], syllables: list[dict]):
    """Assign syllables to notes in order. Holds ('-') become melisma continuations.
    Extra notes repeat the last vowel (melisma); extra syllables are reported."""
    report = {"unassigned_syllables": [], "melisma_notes": 0}
    si = 0
    last = None
    for n in notes:
        n["continuation"] = False
        if si < len(syllables):
            s = syllables[si]
            si += 1
            if s.get("hold") and last is not None:
                n["lyric"] = "-"
                n["phonemes"] = {"onset": [], "vowel": last["vowel"], "coda": []}
                n["continuation"] = True
                report["melisma_notes"] += 1
            else:
                n["lyric"] = s["text"]
                n["phonemes"] = {"onset": s["onset"], "vowel": s["vowel"], "coda": s["coda"]}
                n["ipa"] = s.get("ipa", "")
                last = s
        elif last is not None:
            n["lyric"] = "-"
            n["phonemes"] = {"onset": [], "vowel": last["vowel"], "coda": []}
            n["continuation"] = True
            report["melisma_notes"] += 1
        else:
            n["lyric"] = ""
            n["phonemes"] = {"onset": [], "vowel": "a", "coda": []}
    # move a continuation's coda: the coda belongs at the end of the melisma
    for a, b in zip(notes, notes[1:]):
        if b.get("continuation") and a.get("phonemes") and a["phonemes"]["coda"]:
            b["phonemes"]["coda"], a["phonemes"]["coda"] = a["phonemes"]["coda"], []
    if si < len(syllables):
        report["unassigned_syllables"] = [s["text"] for s in syllables[si:] if not s.get("hold")]
    return notes, report
