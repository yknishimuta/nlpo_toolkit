from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.postprocessing.lemma_normalization_io import (
    load_lemma_normalization_map,
)


def test_loader_preserves_lemma_normalization_tsv_contract(tmp_path: Path) -> None:
    path = tmp_path / "normalization.tsv"
    path.write_text("# comment\n\n arma \t armum \nrosa\trosa\n", encoding="utf-8")
    assert load_lemma_normalization_map(path) == {"arma": "armum", "rosa": "rosa"}


def test_loader_rejects_non_two_column_rows(tmp_path: Path) -> None:
    path = tmp_path / "normalization.tsv"
    path.write_text("arma\tarmum\textra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must have 2 columns"):
        load_lemma_normalization_map(path)
