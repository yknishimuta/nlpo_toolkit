from __future__ import annotations

import pytest

from nlpo_toolkit.corpus_analysis.analysis_records import NLPAnalysisRecord
from nlpo_toolkit.corpus_analysis.features.morphology import (
    MorphologyVocabulary,
    compute_morphology_features,
    encode_morphology_column_component,
)
from nlpo_toolkit.corpus_analysis.features.models import MorphologyOptions
from nlpo_toolkit.corpus_analysis.features.errors import FeatureError
from nlpo_toolkit.nlp.contracts import UDMorphFeature


def _record(*features: UDMorphFeature) -> NLPAnalysisRecord:
    return NLPAnalysisRecord(
        0,
        0,
        0,
        0,
        None,
        None,
        None,
        None,
        "",
        "rosa",
        "rosa",
        "NOUN",
        tuple(features),
    )


def test_morphology_ratios_distinguish_missing_and_unseen_values() -> None:
    vocabulary = MorphologyVocabulary(
        ("Case",),
        (UDMorphFeature("Case", "Acc"), UDMorphFeature("Case", "Nom")),
    )
    values = compute_morphology_features(
        (
            _record(UDMorphFeature("Case", "Nom")),
            _record(UDMorphFeature("Case", "Voc")),
            _record(),
        ),
        vocabulary=vocabulary,
    )
    assert values["morph_coverage_Case"] == pytest.approx(2 / 3)
    assert values["morph_value_Case_Nom"] == pytest.approx(1 / 3)
    assert values["morph_conditional_Case_Nom"] == pytest.approx(1 / 2)
    assert values["morph_other_Case"] == pytest.approx(1 / 3)
    assert values["morph_other_conditional_Case"] == pytest.approx(1 / 2)
    assert "morph_value_Case_Voc" not in values


def test_morphology_encoding_is_ascii_and_collision_free() -> None:
    assert encode_morphology_column_component("Nom") == "Nom"
    assert encode_morphology_column_component("Masc,Neut") == "Masc_u00002c_Neut"
    assert encode_morphology_column_component("a_b") == "a_u00005f_b"
    assert encode_morphology_column_component("æ").isascii()


def test_morphology_options_are_frozen_and_strict() -> None:
    options = MorphologyOptions(True, ["Case", "Number"], 10)  # type: ignore[arg-type]
    assert options.attributes == ("Case", "Number")
    with pytest.raises(FeatureError, match="duplicate"):
        MorphologyOptions(True, ("Case", "Case"))
    with pytest.raises(FeatureError, match="positive integer"):
        MorphologyOptions(True, bundle_top=0)
