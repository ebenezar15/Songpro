"""Small rule-based English letter-to-sound for the local formant engine.
English quality is limited without a pronouncing dictionary; neural engines
(SoulX-Singer) handle English natively."""
from __future__ import annotations

import re

_V = [
    ("igh", "ai"), ("eigh", "eː"), ("ough", "oː"), ("augh", "ɔ"), ("ee", "iː"), ("ea", "iː"), ("oo", "uː"),
    ("ou", "au"), ("ow", "au"), ("oa", "oː"), ("ai", "eː"), ("ay", "eː"), ("oy", "ɔi"), ("oi", "ɔi"),
    ("ie", "iː"), ("ue", "uː"), ("ew", "uː"), ("au", "ɔ"), ("aw", "ɔ"), ("ar", "aː"), ("or", "ɔ"),
    ("er", "ə"), ("ir", "ə"), ("ur", "ə"),
    ("a", "a"), ("e", "e"), ("i", "i"), ("o", "o"), ("u", "a"), ("y", "i"),
]
_C = [
    ("tch", "tʃ"), ("sch", "sk"), ("ch", "tʃ"), ("sh", "ʃ"), ("th", "θ"), ("ph", "f"), ("wh", "w"),
    ("ng", "ŋ"), ("ck", "k"), ("qu", "kw"), ("gh", ""), ("kn", "n"), ("wr", "ɾ"),
    ("b", "b"), ("c", "k"), ("d", "d"), ("f", "f"), ("g", "g"), ("h", "h"), ("j", "dʒ"), ("k", "k"),
    ("l", "l"), ("m", "m"), ("n", "n"), ("p", "p"), ("q", "k"), ("r", "ɾ"), ("s", "s"), ("t", "t"),
    ("v", "v"), ("w", "w"), ("x", "ks"), ("z", "z"),
]


def english_word_to_syllables(word: str):
    word = re.sub(r"[^a-z']", "", word)
    if not word:
        return []
    if len(word) > 3 and word.endswith("e") and word[-2] not in "aeiouy":
        word = word[:-1]  # silent final e
    units = []
    i = 0
    while i < len(word):
        if word[i] == "'":
            i += 1
            continue
        # 'y' at word start is a consonant
        if word[i] == "y" and i == 0:
            units.append(("C", "j", "y"))
            i += 1
            continue
        for pat, ph in _V:
            if word.startswith(pat, i) and not (pat in ("ar", "or", "er", "ir", "ur") and i + 2 < len(word) and word[i + 2] in "aeiou"):
                units.append(("V", ph, pat))
                i += len(pat)
                break
        else:
            for pat, ph in _C:
                if word.startswith(pat, i):
                    for p in re.findall(r"tʃ|dʒ|.", ph) if ph else []:
                        units.append(("C", p, pat))
                    i += len(pat)
                    break
            else:
                i += 1
    sylls, onset, text = [], [], ""
    j = 0
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
        k = j
        while k < len(units) and units[k][0] == "C":
            k += 1
        cl = units[j:k]
        if k >= len(units):
            syl["coda"] = [u[1] for u in cl]
            j = k
        elif len(cl) >= 2:
            syl["coda"] = [u[1] for u in cl[:-1]]
            j = k - 1
        sylls.append(syl)
    if onset:
        if sylls:
            sylls[-1]["coda"] += onset
        else:
            sylls.append({"text": word, "onset": onset[:-1], "vowel": "ə", "coda": onset[-1:]})
    return sylls
