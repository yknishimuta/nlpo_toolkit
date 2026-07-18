from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from nlpo_toolkit.corpus_analysis.analysis_records import (
    AnalysisOptions,
    NLPAnalysisRecord,
    evaluate_analysis_record,
    iter_nlp_analysis_records_from_text,
)
from nlpo_toolkit.corpus_analysis.analysis_policy import AnalysisExtractionPolicy
from nlpo_toolkit.nlp.contracts import NLPDocument, NLPSentence, NLPToken


def _analysis_record(**overrides) -> NLPAnalysisRecord:
    data = {
        "chunk_index": 0,
        "sentence_index": 0,
        "token_index": 0,
        "global_token_index": 0,
        "char_start_in_chunk": 0,
        "char_end_in_chunk": 5,
        "char_start_in_text": 0,
        "char_end_in_text": 5,
        "sentence": "Arma virumque.",
        "token": "Arma",
        "lemma": "arma",
        "upos": "NOUN",
        "morphology": (),
    }
    data.update(overrides)
    return NLPAnalysisRecord(**data)


def _options(**overrides) -> AnalysisOptions:
    data = {
        "group": "text",
        "source_files": (Path("input/text.txt"),),
        "use_lemma": True,
        "upos_targets": frozenset({"NOUN"}),
        "min_token_length": 0,
        "drop_roman_numerals": False,
        "roman_exceptions": frozenset(),
    }
    data.update(overrides)
    return AnalysisOptions(**data)


def test_nlp_analysis_record_fields_are_stable() -> None:
    record = NLPAnalysisRecord(
        chunk_index=0,
        sentence_index=1,
        token_index=2,
        global_token_index=3,
        char_start_in_chunk=4,
        char_end_in_chunk=5,
        char_start_in_text=6,
        char_end_in_text=7,
        sentence="Arma virumque.",
        token="Arma",
        lemma="arma",
        upos="NOUN",
    )

    assert asdict(record) == {
        "chunk_index": 0,
        "sentence_index": 1,
        "token_index": 2,
        "global_token_index": 3,
        "char_start_in_chunk": 4,
        "char_end_in_chunk": 5,
        "char_start_in_text": 6,
        "char_end_in_text": 7,
        "sentence": "Arma virumque.",
        "token": "Arma",
        "lemma": "arma",
        "upos": "NOUN",
        "morphology": (),
    }


def test_evaluate_analysis_record_uses_lemma_key_and_falls_back_to_token() -> None:
    included = evaluate_analysis_record(
        _analysis_record(token="Rosam", lemma="rosa"),
        options=_options(),
    )
    fallback = evaluate_analysis_record(
        _analysis_record(token="Rosam", lemma=None),
        options=_options(),
    )

    assert included.analysis_key == "rosa"
    assert included.included is True
    assert fallback.analysis_key == "rosam"
    assert fallback.included is True


def test_evaluate_analysis_record_surface_mode_uses_surface_key() -> None:
    record = evaluate_analysis_record(
        _analysis_record(token="Rosam", lemma="rosa"),
        options=_options(use_lemma=False),
    )

    assert record.analysis_key == "rosam"
    assert record.included is True


def test_evaluate_analysis_record_upos_and_missing_key_exclusions() -> None:
    upos = evaluate_analysis_record(
        _analysis_record(upos="VERB"),
        options=_options(upos_targets=frozenset({"NOUN"})),
    )
    missing = evaluate_analysis_record(
        _analysis_record(token="", lemma=None),
        options=_options(),
    )

    assert upos.included is False
    assert upos.exclusion_reason == "upos_not_targeted"
    assert missing.included is False
    assert missing.exclusion_reason == "missing_key"


def test_evaluate_analysis_record_minimum_token_length_boundary() -> None:
    equal = evaluate_analysis_record(
        _analysis_record(token="ab", lemma="ab"),
        options=_options(min_token_length=2),
    )
    short = evaluate_analysis_record(
        _analysis_record(token="a", lemma="a"),
        options=_options(min_token_length=2),
    )
    disabled = evaluate_analysis_record(
        _analysis_record(token="a", lemma="a"),
        options=_options(min_token_length=0),
    )

    assert equal.included is True
    assert short.exclusion_reason == "too_short"
    assert disabled.included is True


def test_evaluate_analysis_record_roman_numeral_filter_and_exceptions() -> None:
    excluded = evaluate_analysis_record(
        _analysis_record(token="xiv", lemma="xiv"),
        options=_options(drop_roman_numerals=True),
    )
    configured_exception = evaluate_analysis_record(
        _analysis_record(token="xiv", lemma="xiv"),
        options=_options(
            drop_roman_numerals=True,
            roman_exceptions=frozenset({"xiv"}),
        ),
    )
    drop_disabled = evaluate_analysis_record(
        _analysis_record(token="xiv", lemma="xiv"),
        options=_options(drop_roman_numerals=False),
    )
    surface_builtin_exception = evaluate_analysis_record(
        _analysis_record(token="VI", lemma="vi"),
        options=_options(use_lemma=False, drop_roman_numerals=True),
    )

    assert excluded.exclusion_reason == "roman_numeral"
    assert configured_exception.included is True
    assert drop_disabled.included is True
    assert surface_builtin_exception.included is True


class _FakeBackend:
    def __call__(self, text: str) -> NLPDocument:
        tokens: list[NLPToken] = []
        cursor = 0
        for part in text.split():
            start = text.index(part, cursor)
            end = start + len(part)
            cursor = end
            tokens.append(NLPToken(part, part.lower(), "NOUN", start, end))
        return NLPDocument(sentences=[NLPSentence(tokens=tokens, text=text)])


def test_iter_nlp_analysis_records_from_text_preserves_offsets_and_global_indices() -> (
    None
):
    records = list(
        iter_nlp_analysis_records_from_text(
            text="aa bb cc dd",
            nlp=_FakeBackend(),
            policy=AnalysisExtractionPolicy(chunk_chars=5),
        )
    )

    assert [record.global_token_index for record in records] == list(
        range(len(records))
    )
    assert {record.chunk_index for record in records} == {0, 1, 2}
    assert records[0].sentence == "aa bb"
    assert records[0].token == "aa"
    assert records[0].lemma == "aa"
    assert records[0].upos == "NOUN"
    assert records[0].char_start_in_chunk == 0
    assert records[0].char_start_in_text == 0
    assert records[-1].token == "dd"
    assert records[-1].char_start_in_text == 9
