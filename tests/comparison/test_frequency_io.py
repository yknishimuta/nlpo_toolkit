from pathlib import Path

import pytest

from nlpo_toolkit.comparison.errors import FrequencyTableReadError
from nlpo_toolkit.comparison.frequency_io import (
    derive_frequency_labels, detect_frequency_columns, read_frequency_counts,
)


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_columns_counts_duplicates_and_empty_keys(tmp_path):
    assert detect_frequency_columns(["lemma", "count"]) == ("lemma", "count")
    path = write(tmp_path / "f.csv", "lemma,count\nrosa,2\n,9\nrosa,3\n")
    assert read_frequency_counts(path) == {"rosa": 5.0}


def test_explicit_columns_and_invalid_cell(tmp_path):
    path = write(tmp_path / "f.csv", "word,n\nrosa,2\n")
    assert read_frequency_counts(path, key_column="word", count_column="n") == {"rosa": 2}
    bad = write(tmp_path / "bad.csv", "lemma,count\nrosa,nope\n")
    with pytest.raises(FrequencyTableReadError, match="row 2, column count"):
        read_frequency_counts(bad)


def test_labels_are_deterministic():
    assert derive_frequency_labels((Path("frequency_text.csv"), Path("frequency_text.csv"))) == ("text", "text_2")
