import csv
from pathlib import Path
from types import SimpleNamespace
from collections import Counter

import nlpo_toolkit.nlp as nlp_mod


class DummyWord:
    def __init__(self, text: str, lemma: str, upos: str):
        self.text = text
        self.lemma = lemma
        self.upos = upos


class DummyToken:
    def __init__(self, text: str, start_char: int, words):
        self.text = text
        self.start_char = start_char
        self.words = words  # list[DummyWord]


class DummySent:
    def __init__(self, text: str, tokens):
        self.text = text
        self.tokens = tokens  # list[DummyToken]


class DummyDoc:
    def __init__(self, sentences):
        self.sentences = sentences


def test_trace_writes_offset_columns_and_values(tmp_path):
    # Build a dummy nlp() that returns doc with start_char offsets
    # Text: "Puella rosam amat."
    # "Puella" starts at 0, "rosam" at 7, "amat" at 13 (approx; just needs to be consistent)
    sent_text = "Puella rosam amat."
    doc = DummyDoc(
        sentences=[
            DummySent(
                sent_text,
                tokens=[
                    DummyToken("Puella", 0, [DummyWord("Puella", "puella", "NOUN")]),
                    DummyToken("rosam", 7, [DummyWord("rosam", "rosa", "NOUN")]),
                    DummyToken("amat", 13, [DummyWord("amat", "amo", "VERB")]),
                ],
            )
        ]
    )

    def dummy_nlp(_chunk: str):
        return doc

    out = tmp_path / "trace.tsv"

    # Call trace path directly
    cnt = nlp_mod._count_nouns_streaming_trace(
        text=sent_text,
        nlp=dummy_nlp,
        use_lemma=True,
        upos_targets=frozenset({"NOUN"}),
        chunk_chars=200_000,
        label="",
        trace_tsv=out,
        trace_max_rows=0,
        ref_tag_detector=None,
        ref_tag_counter=None,
    )

    # Counting should include only NOUNs (puella, rosa)
    assert cnt == Counter({"puella": 1, "rosa": 1})

    assert out.exists()

    rows = list(csv.reader(out.open(encoding="utf-8"), delimiter="\t"))
    assert rows, "trace.tsv must not be empty"

    header = rows[0]
    expected_header = [
        "label",
        "chunk",
        "sent_idx",
        "token_idx",
        "token_char_start_in_chunk",
        "token_char_start_in_text",
        "sentence",
        "token",
        "lemma",
        "upos",
        "ref_tag",
        "global_row",
    ]
    assert header == expected_header

    data = rows[1:]
    # We output rows only for NOUN hits => 2 rows expected
    assert len(data) == 2

    # Row for Puella
    r0 = data[0]
    assert r0[7] == "Puella"
    assert r0[8] == "puella"
    assert r0[9] == "NOUN"
    assert r0[4] == "0"   # start in chunk
    assert r0[5] == "0"   # start in text (only one chunk)

    # Row for rosam
    r1 = data[1]
    assert r1[7] == "rosam"
    assert r1[8] == "rosa"
    assert r1[9] == "NOUN"
    assert r1[4] == "7"
    assert r1[5] == "7"


def test_trace_text_offset_increments_across_chunks(tmp_path, monkeypatch):
    # We want to verify token_char_start_in_text = chunk_base_offset + start_char
    # We'll force iter_char_chunks() to yield two chunks, and return different docs per call.

    # chunk1 length = 5
    chunk1 = "aaaaa"
    chunk2 = "bbbbb"
    full_text = chunk1 + chunk2

    # In chunk1: token at start_char=2
    doc1 = DummyDoc(
        sentences=[DummySent("S1", tokens=[DummyToken("X", 2, [DummyWord("X", "x", "NOUN")])])]
    )
    # In chunk2: token at start_char=1 -> absolute should be 5 + 1 = 6
    doc2 = DummyDoc(
        sentences=[DummySent("S2", tokens=[DummyToken("Y", 1, [DummyWord("Y", "y", "NOUN")])])]
    )

    calls = {"n": 0}

    def dummy_nlp(_chunk: str):
        calls["n"] += 1
        return doc1 if calls["n"] == 1 else doc2

    # Monkeypatch chunker to yield our fixed chunks
    monkeypatch.setattr(nlp_mod, "iter_char_chunks", lambda _t, chunk_chars=200_000: iter([chunk1, chunk2]))

    out = tmp_path / "trace.tsv"
    cnt = nlp_mod._count_nouns_streaming_trace(
        text=full_text,
        nlp=dummy_nlp,
        use_lemma=True,
        upos_targets=frozenset({"NOUN"}),
        chunk_chars=200_000,
        label="",
        trace_tsv=out,
        trace_max_rows=0,
        ref_tag_detector=None,
        ref_tag_counter=None,
    )

    assert cnt == Counter({"x": 1, "y": 1})

    rows = list(csv.reader(out.open(encoding="utf-8"), delimiter="\t"))
    data = rows[1:]
    assert len(data) == 2

    # First row absolute offset should be 2
    assert data[0][4] == "2"
    assert data[0][5] == "2"

    # Second row absolute offset should be 6 (= len(chunk1)=5 + start_char=1)
    assert data[1][4] == "1"
    assert data[1][5] == "6"
