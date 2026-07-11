import pytest
from types import SimpleNamespace

from nlpo_toolkit.corpus_analysis.config import NormalizationConfig
from nlpo_toolkit.corpus_analysis.normalizer import normalize_text


@pytest.mark.parametrize(
    ("config", "text", "expected"),
    [
        (NormalizationConfig(enabled=False, casefold=True), "VIVIT", "VIVIT"),
        (NormalizationConfig(unicode_nf="NFKD"), "ﬃ", "ffi"),
        (NormalizationConfig(normalize_ligatures=True), "æneas œ", "aeneas oe"),
        (NormalizationConfig(strip_diacritics=True), "múltās", "multas"),
        (NormalizationConfig(map_u_v=True), "Vivit", "Uiuit"),
        (NormalizationConfig(map_i_j=True), "Julius", "Iulius"),
        (NormalizationConfig(casefold=True), "ÆNEAS", "æneas"),
    ],
)
def test_active_normalization_options_still_apply(
    config: NormalizationConfig,
    text: str,
    expected: str,
) -> None:
    assert normalize_text(text, SimpleNamespace(normalization=config)) == expected


def test_active_normalization_options_still_combine_in_current_order() -> None:
    config = NormalizationConfig(
        enabled=True,
        unicode_nf="NFKD",
        normalize_ligatures=True,
        strip_diacritics=True,
        map_u_v=True,
        map_i_j=True,
        casefold=True,
    )

    assert normalize_text(
        "ÆNEAS VIVIT JÚLIA",
        SimpleNamespace(normalization=config),
    ) == "aeneas uiuit iulia"
