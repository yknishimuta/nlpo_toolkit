from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

from nlpo_toolkit.corpus_analysis.analysis_records import NLPAnalysisRecord
from nlpo_toolkit.corpus_analysis.corpus import PreparedCorpus
from nlpo_toolkit.corpus_analysis.features.engine import (
    analyze_feature_corpora,
    build_feature_rows,
    fit_feature_vocabulary,
)
from nlpo_toolkit.corpus_analysis.features.models import (
    AnalyzedFeatureCorpus,
    CharacterNgramOptions,
    FeatureOptions,
    UposNgramOptions,
    FeatureFilterPolicy,
    MorphologyOptions,
)
from nlpo_toolkit.corpus_analysis.analysis_policy import AnalysisExtractionPolicy
from nlpo_toolkit.nlp.contracts import NLPDocument, NLPSentence, NLPToken
from nlpo_toolkit.nlp.contracts import UDMorphFeature


def _record(token: str, upos: str, index: int) -> NLPAnalysisRecord:
    return NLPAnalysisRecord(
        0, 0, index, index, None, None, None, None, "", token, token, upos
    )


def _analyzed(label: str, text: str, tokens: tuple[tuple[str, str], ...]):
    records = tuple(
        _record(token, upos, index) for index, (token, upos) in enumerate(tokens)
    )
    source = PreparedCorpus(label, (Path(f"{label}.txt"),), text, text, Counter())
    return AnalyzedFeatureCorpus(source, records, records, text=text)


def test_training_only_fit_excludes_held_out_mfw_character_and_upos_terms() -> None:
    training = (
        _analyzed(
            "a1", "et et abcabc", (("et", "NOUN"), ("et", "VERB"), ("et", "NOUN"))
        ),
        _analyzed("b1", "et abcabc", (("et", "NOUN"), ("et", "VERB"))),
    )
    held_out = _analyzed("held", "zzzzzzzzzzzz", tuple(("zzzz", "X") for _ in range(8)))
    options = FeatureOptions(
        mfw=1,
        field="token",
        character_ngrams=CharacterNgramOptions((3,), top=2),
        upos_ngrams=UposNgramOptions((2,), top=1),
    )
    vocabulary = fit_feature_vocabulary(training, options=options)

    assert vocabulary.mfw_terms == ("et",)
    assert "zzzz" not in vocabulary.mfw_terms
    assert all(term.value != "zzz" for term in vocabulary.character_ngrams.terms)
    assert all(term.tags != ("X", "X") for term in vocabulary.upos_ngrams.terms)

    rows = build_feature_rows(
        training + (held_out,), options=options, vocabulary=vocabulary
    )
    assert tuple(rows[-1]) == tuple(rows[0])
    assert rows[-1]["mfw_et"] == 0.0


def test_fitted_vocabulary_can_differ_by_held_out_work() -> None:
    corpora = (
        _analyzed("a", "aaaa", (("a", "NOUN"), ("a", "VERB"))),
        _analyzed("b", "bbbb", (("b", "ADJ"), ("b", "NOUN"))),
        _analyzed("c", "cccc", (("c", "VERB"), ("c", "ADJ"))),
    )
    options = FeatureOptions(mfw=1, field="token")
    without_a = fit_feature_vocabulary(corpora[1:], options=options)
    without_c = fit_feature_vocabulary(corpora[:2], options=options)
    assert without_a.mfw_terms != without_c.mfw_terms


def test_training_only_morphology_vocabulary_routes_held_out_value_to_other() -> None:
    training_corpus = _analyzed("train", "rosa", (("rosa", "NOUN"),))
    held_corpus = _analyzed("held", "rosa", (("rosa", "NOUN"),))
    training_record = replace(
        training_corpus.lexical_records[0],
        morphology=(UDMorphFeature("Case", "Nom"),),
    )
    held_record = replace(
        held_corpus.lexical_records[0],
        morphology=(UDMorphFeature("Case", "Voc"),),
    )
    training = replace(
        training_corpus,
        raw_records=(training_record,),
        lexical_records=(training_record,),
    )
    held = replace(
        held_corpus, raw_records=(held_record,), lexical_records=(held_record,)
    )
    options = FeatureOptions(
        include_basic=False,
        include_upos=False,
        morphology=MorphologyOptions(True, ("Case",)),
    )
    vocabulary = fit_feature_vocabulary((training,), options=options)
    assert vocabulary.morphology.values == (UDMorphFeature("Case", "Nom"),)
    row = build_feature_rows((held,), options=options, vocabulary=vocabulary)[0]
    assert "morph_value_Case_Voc" not in row
    assert row["morph_other_Case"] == 1.0
    assert row["morph_other_conditional_Case"] == 1.0


def test_all_corpora_are_analyzed_once_before_fold_vocabulary_fit() -> None:
    calls: list[str] = []

    def nlp(text: str) -> NLPDocument:
        calls.append(text)
        return NLPDocument((NLPSentence((NLPToken(text, text, "NOUN"),), text),), text)

    corpora = tuple(
        PreparedCorpus(label, (Path(f"{label}.txt"),), label, label, Counter())
        for label in ("a1", "a2", "b1", "b2")
    )
    analyzed = analyze_feature_corpora(
        corpora,
        nlp=nlp,
        extraction_policy=AnalysisExtractionPolicy(),
        filter_policy=FeatureFilterPolicy(),
    )
    for held_out in analyzed:
        fit_feature_vocabulary(
            tuple(item for item in analyzed if item is not held_out),
            options=FeatureOptions(mfw=1),
        )
    assert calls == ["a1", "a2", "b1", "b2"]
