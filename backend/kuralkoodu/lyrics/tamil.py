"""Tamil grapheme-to-phoneme for singing.

Handles Tamil script (தமிழ்) and romanised "Tanglish". Implements the contextual
allophony that makes Tamil sound Tamil rather than "Tamil read by an English model":

* The five stops க ச ட த ப are written once but pronounced by context:
    - word-initial / geminate (க்க):  voiceless unaspirated   k  tʃ/s  ʈ  t̪  p
    - after a nasal (ங்க, ஞ்ச, ண்ட, ந்த, ம்ப): voiced           g  dʒ    ɖ  d̪  b
    - between vowels:                  lenited                 ɣ~h s   ɖ  ð  β
  Tamil stops are never aspirated (an English 'k' puff of air sounds foreign).
* ழ is the retroflex approximant ɻ ("zha"), ள = ɭ, ண = ɳ  (F3 lowered in synthesis).
* ர = tap ɾ, ற = trill r; ற்ற -> "tr" (வெற்றி = vetri); ன்ற -> "ndr" (ஒன்று = ondru).
* ந = dental n̪, ன = alveolar n.
* Word-final short உ after a consonant is the unrounded ɨ (kuṟṟiyalukaram): அது = adɨ.
* Optional colloquial glides: word-initial எ/ஏ -> je (and ஒ/ஓ -> wo with glides='full').
* Dialect: word-initial ச = s (Chennai / film standard) or tʃ (literary / Sri Lankan).
"""
from __future__ import annotations

import re
import unicodedata

VOWELS_INDEP = {
    "அ": "a", "ஆ": "aː", "இ": "i", "ஈ": "iː", "உ": "u", "ஊ": "uː",
    "எ": "e", "ஏ": "eː", "ஐ": "ai", "ஒ": "o", "ஓ": "oː", "ஔ": "au",
}
VOWEL_SIGNS = {
    "ா": "aː", "ி": "i", "ீ": "iː", "ு": "u", "ூ": "uː",
    "ெ": "e", "ே": "eː", "ை": "ai", "ொ": "o", "ோ": "oː", "ௌ": "au",
}
PULLI = "்"
AYTHAM = "ஃ"
# base consonant -> internal letter key
CONSONANTS = {
    "க": "K", "ங": "ŋ", "ச": "C", "ஞ": "ɲ", "ட": "T", "ண": "ɳ", "த": "D", "ந": "n̪",
    "ப": "P", "ம": "m", "ய": "j", "ர": "ɾ", "ல": "l", "வ": "ʋ", "ழ": "ɻ", "ள": "ɭ",
    "ற": "R", "ன": "n", "ஜ": "dʒ", "ஶ": "ʃ", "ஷ": "ʂ", "ஸ": "s", "ஹ": "h",
}
STOPS = {"K", "C", "T", "D", "P"}
NASALS = {"ŋ", "ɲ", "ɳ", "n̪", "n", "m"}
_VOICELESS = {"K": "k", "C": "tʃ", "T": "ʈ", "D": "t̪", "P": "p"}
_VOICED = {"K": "g", "C": "dʒ", "T": "ɖ", "D": "d̪", "P": "b"}
_LENITED = {"K": "ɣ", "C": "s", "T": "ɖ", "D": "ð", "P": "β"}

TAMIL_RE = re.compile(r"[஀-௿]")


def is_tamil_script(text: str) -> bool:
    return bool(TAMIL_RE.search(text))


def _letters(word: str):
    """Split a Tamil word into (consonant_key|None, vowel|None|'∅', grapheme) units."""
    out = []
    i = 0
    w = unicodedata.normalize("NFC", word)
    # NFC leaves some two-part vowel signs decomposed in older text; fix them
    w = w.replace("ொ", "ொ").replace("ோ", "ோ").replace("ௌ", "ௌ")
    while i < len(w):
        ch = w[i]
        if ch in VOWELS_INDEP:
            out.append((None, VOWELS_INDEP[ch], ch))
            i += 1
        elif ch == AYTHAM:
            out.append(("ஃ", "∅", ch))
            i += 1
        elif ch in CONSONANTS:
            nxt = w[i + 1] if i + 1 < len(w) else ""
            # ஶ்ரீ / க்ஷ handled naturally as clusters
            if nxt == PULLI:
                out.append((CONSONANTS[ch], "∅", ch + nxt))
                i += 2
            elif nxt in VOWEL_SIGNS:
                out.append((CONSONANTS[ch], VOWEL_SIGNS[nxt], ch + nxt))
                i += 2
            else:
                out.append((CONSONANTS[ch], "a", ch))
                i += 1
        else:
            i += 1  # ignore punctuation / ZWJ etc.
    return out


def _stop_allophone(key, prev_cons, prev_is_vowel, word_initial, dialect):
    if prev_cons == key:  # geminate: second half voiceless
        return _VOICELESS[key]
    if prev_cons in NASALS:
        return _VOICED[key]
    if word_initial:
        if key == "C":
            return "s" if dialect == "standard" else "tʃ"
        return _VOICELESS[key]
    if prev_is_vowel:
        return _LENITED[key]
    if prev_cons == "ஃ":
        return {"K": "x", "P": "f"}.get(key, _VOICELESS[key])
    return _VOICELESS[key]  # after ய ர ல ழ ள etc.


def tamil_word_to_syllables(word: str, dialect="standard", glides=True):
    letters = _letters(word)
    if not letters:
        return []
    # 1) phoneme per letter
    phon = []  # list of dict(cons=[...], vowel=str|None, g=grapheme)
    for idx, (c, v, g) in enumerate(letters):
        prev = letters[idx - 1] if idx > 0 else None
        prev_cons = prev[0] if prev and prev[1] == "∅" else None
        prev_is_vowel = prev is not None and prev[1] not in ("∅", None)
        word_initial = idx == 0
        cons = []
        if c is None:
            pass
        elif c == "ஃ":
            cons = []
        elif c in STOPS and v == "∅":
            # a dead stop (coda / first half of a geminate) is voiceless, except after a nasal
            cons = [_VOICED[c] if prev_cons in NASALS else _VOICELESS[c]]
        elif c in STOPS:
            cons = [_stop_allophone(c, prev_cons, prev_is_vowel, word_initial, dialect)]
        elif c == "R":
            if prev_cons == "R":
                cons = ["r"]  # second half of ற்ற  -> t + r
            else:
                cons = ["r"]  # ன்ற: the d is inserted on the coda side below
        else:
            cons = [c]
        phon.append({"cons": cons, "vowel": v, "g": g, "key": c})

    # coda-side rewrites for clusters
    for idx, p in enumerate(phon):
        nxt = phon[idx + 1] if idx + 1 < len(phon) else None
        if p["vowel"] == "∅" and p["key"] == "R" and nxt and nxt["key"] == "R":
            p["cons"] = ["t"]  # ற்ற -> t r   (வெற்றி vetri)
        if p["vowel"] == "∅" and p["key"] == "n" and nxt and nxt["key"] == "R":
            p["cons"] = ["n", "d"]  # ன்ற -> n d r (ஒன்று ondru)
        if p["key"] == "ஃ":
            p["cons"] = []

    # word-final kuṟṟiyalukaram
    last = phon[-1]
    if len(phon) > 1 and last["vowel"] == "u" and last["key"] is not None:
        last["vowel"] = "ɨ"
    # colloquial initial glides
    first = phon[0]
    if glides and first["key"] is None and first["vowel"] in ("e", "eː"):
        first["cons"] = ["j"]
    if glides == "full" and first["key"] is None and first["vowel"] in ("o", "oː"):
        first["cons"] = ["w"]

    # 2) syllabify: dead consonants become codas of the preceding syllable
    sylls = []
    pending_onset = []
    pending_g = ""
    for p in phon:
        if p["vowel"] == "∅":
            if sylls:
                sylls[-1]["coda"] += p["cons"]
                sylls[-1]["text"] += p["g"]
            else:
                pending_onset += p["cons"]
                pending_g += p["g"]
            continue
        sylls.append({"text": pending_g + p["g"], "onset": pending_onset + p["cons"], "vowel": p["vowel"], "coda": []})
        pending_onset, pending_g = [], ""
    if pending_onset and sylls:
        sylls[-1]["coda"] += pending_onset
    return sylls


# ---------------------------------------------------------------- romanised
_ROMAN = [
    ("aa", "V", "aː"), ("ee", "V", "iː"), ("ii", "V", "iː"), ("oo", "V", "oː"), ("uu", "V", "uː"),
    ("ai", "V", "ai"), ("au", "V", "au"), ("ei", "V", "eː"),
    ("A", "V", "aː"), ("I", "V", "iː"), ("U", "V", "uː"), ("E", "V", "eː"), ("O", "V", "oː"),
    ("a", "V", "a"), ("i", "V", "i"), ("u", "V", "u"), ("e", "V", "e"), ("o", "V", "o"),
    ("zh", "C", "ɻ"), ("ch", "C", "tʃ"), ("sh", "C", "ʂ"), ("th", "C", "t̪"), ("dh", "C", "ð"),
    ("ng", "C", "ŋ"), ("nj", "C", "ɲ"), ("ny", "C", "ɲ"), ("ksh", "C", "kʂ"),
    ("rr", "C", "r"), ("R", "C", "r"), ("L", "C", "ɭ"), ("N", "C", "ɳ"), ("T", "C", "ʈ"), ("D", "C", "ɖ"),
    ("k", "C", "k"), ("g", "C", "g"), ("c", "C", "tʃ"), ("j", "C", "dʒ"), ("s", "C", "s"), ("t", "C", "ʈ"),
    ("d", "C", "ɖ"), ("n", "C", "n"), ("p", "C", "p"), ("b", "C", "b"), ("m", "C", "m"), ("y", "C", "j"),
    ("r", "C", "ɾ"), ("l", "C", "l"), ("v", "C", "ʋ"), ("w", "C", "ʋ"), ("f", "C", "f"), ("h", "C", "h"),
    ("z", "C", "z"), ("x", "C", "ks"), ("q", "C", "k"),
]


def roman_word_to_syllables(word: str):
    units = []
    i = 0
    while i < len(word):
        for pat, kind, ph in _ROMAN:
            if word.startswith(pat, i):
                units.append((kind, ph, pat))
                i += len(pat)
                break
        else:
            i += 1
    sylls = []
    j = 0
    onset, text = [], ""
    while j < len(units):
        kind, ph, pat = units[j]
        if kind == "C":
            onset.append(ph)
            text += pat
            j += 1
            continue
        syl = {"text": text + pat, "onset": onset, "vowel": ph, "coda": []}
        onset, text = [], ""
        j += 1
        # consonants until next vowel: all but the last go to coda (maximal single onset)
        k = j
        while k < len(units) and units[k][0] == "C":
            k += 1
        cluster = units[j:k]
        if k >= len(units):  # word end: all coda
            syl["coda"] = [u[1] for u in cluster]
            syl["text"] += "".join(u[2] for u in cluster)
            j = k
        elif len(cluster) >= 2:
            syl["coda"] = [u[1] for u in cluster[:-1]]
            syl["text"] += "".join(u[2] for u in cluster[:-1])
            j = k - 1
        sylls.append(syl)
    if onset and sylls:
        sylls[-1]["coda"] += onset
    # final short u -> ɨ (Tamil habit), only if multisyllabic
    if len(sylls) > 1 and sylls[-1]["vowel"] == "u" and sylls[-1]["onset"]:
        sylls[-1]["vowel"] = "ɨ"
    return sylls
