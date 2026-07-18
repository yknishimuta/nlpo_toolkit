from __future__ import annotations

from types import SimpleNamespace

import pytest

from nlpo_toolkit.backends.stanza_backend import (
    StanzaBackendDataError,
    convert_stanza_document_to_common_model,
    parse_stanza_feats,
)
from nlpo_toolkit.nlp.contracts import NLPToken, UDMorphFeature
from nlpo_toolkit.corpus_analysis.analysis_records import (
    AnalysisOptions,
    evaluate_analysis_record,
    iter_nlp_analysis_records,
)
from nlpo_toolkit.nlp.contracts import NLPDocument, NLPSentence


@pytest.mark.parametrize("raw", (None, "", "_"))
def test_empty_stanza_feats(raw) -> None:
    assert parse_stanza_feats(raw) == ()


def test_stanza_feats_are_typed_and_canonical() -> None:
    expected = (
        UDMorphFeature("Case", "Nom"),
        UDMorphFeature("Gender", "Fem"),
        UDMorphFeature("Number", "Sing"),
    )
    assert parse_stanza_feats("Number=Sing|Case=Nom|Gender=Fem") == expected
    word = SimpleNamespace(
        text="rosa",
        lemma="rosa",
        upos="NOUN",
        start_char=0,
        end_char=4,
        feats="Case=Nom|Number=Sing",
    )
    document = convert_stanza_document_to_common_model(
        SimpleNamespace(sentences=[SimpleNamespace(words=[word], text="rosa")]),
        "rosa",
    )
    assert document.sentences[0].tokens[0].morphology == (
        UDMorphFeature("Case", "Nom"),
        UDMorphFeature("Number", "Sing"),
    )


@pytest.mark.parametrize(
    "raw",
    (
        1,
        "Case",
        "=Nom",
        "Case=",
        "Case=Nom|Case=Acc",
        "Case=Nom||Number=Sing",
        "Case=Nom=Extra",
    ),
)
def test_invalid_stanza_feats_are_rejected(raw) -> None:
    with pytest.raises(StanzaBackendDataError):
        parse_stanza_feats(raw)


def test_nlp_token_positional_offsets_remain_compatible() -> None:
    token = NLPToken("rosa", "rosa", "NOUN", 0, 4)
    assert (token.start_char, token.end_char, token.morphology) == (0, 4, ())


def test_morphology_propagates_to_included_and_excluded_records() -> None:
    feature = UDMorphFeature("Case", "Nom")
    document = NLPDocument(
        (NLPSentence((NLPToken("rosa", "rosa", "NOUN", morphology=(feature,)),)),)
    )
    analysis = tuple(
        iter_nlp_analysis_records(
            document=document,
            chunk_index=0,
            chunk_start_in_text=0,
            global_token_start=0,
        )
    )[0]
    included = evaluate_analysis_record(
        analysis,
        options=AnalysisOptions("g", (), True, frozenset(("NOUN",))),
    )
    excluded = evaluate_analysis_record(
        analysis,
        options=AnalysisOptions("g", (), True, frozenset(("VERB",))),
    )
    assert (
        analysis.morphology == included.morphology == excluded.morphology == (feature,)
    )
    assert included.included is True
    assert excluded.included is False
