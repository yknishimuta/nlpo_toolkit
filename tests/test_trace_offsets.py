import csv
from pathlib import Path
from collections import Counter

import nlpo_toolkit.nlp as nlp_mod
from nlpo_toolkit.models import NLPDocument, NLPSentence, NLPToken


def test_trace_writes_offset_columns_and_values(tmp_path):
    sent_text = "Puella rosam amat."
    
    # Create mock directly using the common data model
    doc = NLPDocument(
        sentences=[
            NLPSentence(
                text=sent_text,
                tokens=[
                    NLPToken(text="Puella", lemma="puella", upos="NOUN", start_char=0),
                    NLPToken(text="rosam", lemma="rosa", upos="NOUN", start_char=7),
                    NLPToken(text="amat", lemma="amo", upos="VERB", start_char=13),
                ],
            )
        ]
    )

    def dummy_nlp(_chunk: str) -> NLPDocument:
        return doc

    out = tmp_path / "trace.tsv"

    cnt = nlp_mod._count_nouns_streaming_trace(
        text=sent_text,
        nlp=dummy_nlp,
        use_lemma=True,
        upos_targets=frozenset({"NOUN"}),
        chunk_chars=200_000,
        label="",
        trace_tsv=out,
        trace_max_rows=0,
        trace_only_keys=None,
        trace_write_truncation_marker=True,
        ref_tag_detector=None,
        ref_tag_counter=None,
    )

    # Verify that only NOUNs are counted
    assert cnt == Counter({"puella": 1, "rosa": 1})
    assert out.exists()

    rows = list(csv.reader(out.open(encoding="utf-8"), delimiter="\t"))
    assert rows, "trace.tsv must not be empty"

    header = rows[0]
    expected_header = [
        "label", "chunk", "sent_idx", "token_idx",
        "token_char_start_in_chunk", "token_char_start_in_text",
        "sentence", "token", "lemma", "upos", "ref_tag", "global_row",
    ]
    assert header == expected_header

    data = rows[1:]
    assert len(data) == 2

    # Row for Puella
    r0 = data[0]
    assert r0[7] == "Puella"
    assert r0[8] == "puella"
    assert r0[9] == "NOUN"
    assert r0[4] == "0"   # start in chunk
    assert r0[5] == "0"   # start in text

    # Row for rosam
    r1 = data[1]
    assert r1[7] == "rosam"
    assert r1[8] == "rosa"
    assert r1[9] == "NOUN"
    assert r1[4] == "7"   # start in chunk
    assert r1[5] == "7"   # start in text


def test_trace_text_offset_increments_across_chunks(tmp_path, monkeypatch):
    chunk1 = "aaaaa"
    chunk2 = "bbbbb"
    full_text = chunk1 + chunk2

    # Configure to return different NLPDocuments for each chunk
    doc1 = NLPDocument(
        sentences=[NLPSentence(text="S1", tokens=[NLPToken(text="X", lemma="x", upos="NOUN", start_char=2)])]
    )
    doc2 = NLPDocument(
        sentences=[NLPSentence(text="S2", tokens=[NLPToken(text="Y", lemma="y", upos="NOUN", start_char=1)])]
    )

    calls = {"n": 0}

    def dummy_nlp(_chunk: str) -> NLPDocument:
        calls["n"] += 1
        return doc1 if calls["n"] == 1 else doc2

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
        trace_only_keys=None,
        trace_write_truncation_marker=True,
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