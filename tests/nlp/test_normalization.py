from __future__ import annotations

from nlpo_toolkit.nlp.normalization import normalize_token


def test_empty_case_and_ligatures() -> None:
    assert normalize_token("") == ""
    assert normalize_token("ÆNEAS ŒDIPUS") == "aeneas oedipus"
    assert normalize_token("ROMA", lower=False) == "ROMA"


def test_diacritics_can_be_removed_or_preserved() -> None:
    assert normalize_token("rōsa") == "rosa"
    assert normalize_token("rōsa", strip_diacritics=False) == "rōsa"


def test_custom_and_empty_ligature_maps() -> None:
    assert normalize_token("x", ligature_map={"x": "ks"}) == "ks"
    assert normalize_token("æ", ligature_map={}) == "æ"


def test_input_value_is_unchanged() -> None:
    source = "ÆNĒAS"
    assert normalize_token(source) == "aeneas"
    assert source == "ÆNĒAS"
