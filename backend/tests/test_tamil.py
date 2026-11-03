import pytest

from kuralkoodu.lyrics import align_to_notes, parse_lyrics


def ipa(word, **kw):
    return [s["ipa"] for s in parse_lyrics(word, "ta", **kw) if not s.get("hold")]


@pytest.mark.parametrize("word,expected", [
    ("அது", ["a", "ðɨ"]),                 # intervocalic த -> ð, final u -> ɨ
    ("வெற்றி", ["ʋet", "ri"]),            # ற்ற -> tr
    ("ஒன்று", ["ond", "rɨ"]),             # ன்ற -> ndr
    ("தமிழ்", ["t̪a", "miɻ"]),            # ழ -> ɻ
    ("அன்பே", ["an", "beː"]),             # post-nasal voicing
    ("முத்தம்", ["mut̪", "t̪am"]),          # geminate voiceless
    ("பாடல்", ["paː", "ɖal"]),            # intervocalic ட -> ɖ
    ("கண்ணே", ["kaɳ", "ɳeː"]),           # retroflex nasal
    ("எங்கே", ["jeŋ", "geː"]),           # colloquial glide + ŋk -> ŋg
])
def test_tamil_allophony(word, expected):
    assert ipa(word) == expected


def test_dialect_and_glides():
    assert ipa("சந்தோஷம்")[0].startswith("s")
    assert ipa("சந்தோஷம்", dialect="literary")[0].startswith("tʃ")
    assert ipa("எங்கே", glides=False)[0] == "eŋ"


def test_romanised_tamil():
    assert [s["ipa"] for s in parse_lyrics("thamizh", "ta-Latn")] == ["t̪a", "miɻ"]


def test_alignment_with_melisma():
    notes = [{"id": f"n{i}", "start": i * 0.5, "dur": 0.45, "midi": 60} for i in range(5)]
    syl = parse_lyrics("நிலா - வே", "ta")
    out, rep = align_to_notes(notes, syl)
    assert [n["lyric"] for n in out] == ["நி", "லா", "-", "வே", "-"]
    assert out[2]["continuation"] and out[2]["phonemes"]["vowel"] == "aː"
    assert rep["melisma_notes"] == 2
