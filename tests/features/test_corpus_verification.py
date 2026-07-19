from __future__ import annotations

from collections import Counter
from pathlib import Path

from nlpo_toolkit.corpus_analysis.analysis_records import NLPAnalysisRecord
from nlpo_toolkit.corpus_analysis.corpus import PreparedCorpus
from nlpo_toolkit.corpus_analysis.features.corpus_stylometry_support import (
    PreparedCorpusStylometryData,
)
from nlpo_toolkit.corpus_analysis.features.corpus_verification_models import (
    CorpusVerificationRequest,
)
from nlpo_toolkit.corpus_analysis.features.corpus_verification_service import (
    CorpusVerificationDependencies,
    execute_corpus_verification,
)
from nlpo_toolkit.corpus_analysis.features.models import (
    AnalyzedFeatureCorpus,
    CharacterNgramOptions,
    FeatureOptions,
    FeatureRequest,
    MorphologyOptions,
    UposNgramOptions,
)
from nlpo_toolkit.corpus_analysis.requests import CorpusPreparationRequest
from nlpo_toolkit.nlp.contracts import UDMorphFeature
from nlpo_toolkit.stylometry.evaluation_models import AuthorshipMetadata


def _corpus(
    label: str,
    tokens: tuple[str, ...],
    upos: tuple[str, ...],
    case: str,
) -> AnalyzedFeatureCorpus:
    text = " ".join(tokens)
    records = []
    offset = 0
    for index, (token, tag) in enumerate(zip(tokens, upos, strict=True)):
        records.append(
            NLPAnalysisRecord(
                0,
                0,
                index,
                index,
                offset,
                offset + len(token),
                offset,
                offset + len(token),
                text,
                token,
                token,
                tag,
                (UDMorphFeature("Case", case),),
            )
        )
        offset += len(token) + 1
    source = PreparedCorpus(
        label, (Path(f"{label}.txt"),), text, text, Counter()
    )
    return AnalyzedFeatureCorpus(source, tuple(records), tuple(records), text=text)


def test_query_is_excluded_from_every_fitted_feature_vocabulary(monkeypatch) -> None:
    import nlpo_toolkit.corpus_analysis.features.corpus_verification_service as service

    analyzed = (
        _corpus("ref_a", ("et", "et", "arma"), ("NOUN", "VERB", "NOUN"), "Nom"),
        _corpus("ref_b", ("et", "virum", "et"), ("NOUN", "VERB", "NOUN"), "Acc"),
        _corpus("query", ("zzzz", "zzzz", "zzzz"), ("X", "ADJ", "X"), "Voc"),
    )
    options = FeatureOptions(
        mfw=1,
        include_upos=False,
        include_basic=True,
        character_ngrams=CharacterNgramOptions((3,), 4),
        upos_ngrams=UposNgramOptions((2,), 2),
        morphology=MorphologyOptions(True, ("Case",), 2),
    )
    assignments = {
        "ref_a": ("candidate", "a"),
        "ref_b": ("background", "b"),
        "query": ("unknown", "q"),
    }
    prepared = PreparedCorpusStylometryData(
        analyzed, options, assignments, tuple(assignments)
    )
    monkeypatch.setattr(
        service, "prepare_corpus_stylometry_data", lambda *args, **kwargs: prepared
    )
    sentinel = object()
    monkeypatch.setattr(service, "evaluate_verification", lambda *args, **kwargs: sentinel)
    request = CorpusVerificationRequest(
        FeatureRequest(
            CorpusPreparationRequest(Path("."), Path("config.yml")),
            mfw=1,
        ),
        Path("metadata.csv"),
        "candidate",
        "q",
    )
    result = execute_corpus_verification(
        request,
        dependencies=CorpusVerificationDependencies(
            object(),  # type: ignore[arg-type]
            lambda *args, **kwargs: AuthorshipMetadata(()),
        ),
    )
    assert result.verification is sentinel
    assert result.vocabulary.mfw_terms == ("et",)
    assert all("zzz" not in term.value for term in result.vocabulary.character_ngrams)
    assert all("X" not in term.tags and "ADJ" not in term.tags for term in result.vocabulary.upos_ngrams)
    assert UDMorphFeature("Case", "Voc") not in result.vocabulary.morphology.values
    assert all(
        UDMorphFeature("Case", "Voc") not in bundle.features
        for bundle in result.vocabulary.morphology.bundles
    )


def test_query_changes_do_not_change_vocabulary_calibration_or_thresholds(
    monkeypatch,
) -> None:
    import nlpo_toolkit.corpus_analysis.features.corpus_verification_service as service

    references = (
        _corpus("a1", ("et", "arma", "cano"), ("NOUN", "VERB", "NOUN"), "Nom"),
        _corpus("a2", ("et", "virum", "cano"), ("NOUN", "VERB", "NOUN"), "Acc"),
        _corpus("a3", ("et", "arma", "virum"), ("NOUN", "NOUN", "VERB"), "Nom"),
        _corpus("b1", ("longe", "alia", "verba"), ("ADJ", "NOUN", "NOUN"), "Abl"),
        _corpus("b2", ("diversa", "multa", "sunt"), ("ADJ", "ADJ", "VERB"), "Dat"),
    )
    assignments = {
        "a1": ("A", "A1"),
        "a2": ("A", "A2"),
        "a3": ("A", "A3"),
        "b1": ("B", "B1"),
        "b2": ("C", "B2"),
        "query": ("unknown", "Q"),
    }
    options = FeatureOptions(
        mfw=2,
        include_upos=True,
        include_basic=True,
        character_ngrams=CharacterNgramOptions((3,), 5),
        upos_ngrams=UposNgramOptions((2,), 3),
        morphology=MorphologyOptions(True, ("Case",), 3),
    )
    current = [
        PreparedCorpusStylometryData(
            references
            + (_corpus("query", ("zzzz", "zzzz", "zzzz"), ("X", "X", "X"), "Voc"),),
            options,
            assignments,
            tuple(assignments),
        )
    ]
    monkeypatch.setattr(
        service,
        "prepare_corpus_stylometry_data",
        lambda *args, **kwargs: current[0],
    )
    request = CorpusVerificationRequest(
        FeatureRequest(CorpusPreparationRequest(Path("."), Path("config.yml"))),
        Path("metadata.csv"),
        "A",
        "Q",
    )
    dependencies = CorpusVerificationDependencies(
        object(),  # type: ignore[arg-type]
        lambda *args, **kwargs: AuthorshipMetadata(()),
    )
    first = execute_corpus_verification(request, dependencies=dependencies)
    current[0] = PreparedCorpusStylometryData(
        references
        + (_corpus("query", ("yyyy", "extra", "longissima"), ("ADV", "ADJ", "VERB"), "Voc"),),
        options,
        assignments,
        tuple(assignments),
    )
    second = execute_corpus_verification(request, dependencies=dependencies)

    assert first.vocabulary.sha256 == second.vocabulary.sha256
    assert first.vocabulary == second.vocabulary
    assert first.verification.calibration_scores == second.verification.calibration_scores
    assert first.verification.thresholds == second.verification.thresholds
    assert first.verification.retained_feature_names == second.verification.retained_feature_names
    assert (
        first.verification.dropped_zero_variance_features
        == second.verification.dropped_zero_variance_features
    )
    assert first.verification.query_distance != second.verification.query_distance
