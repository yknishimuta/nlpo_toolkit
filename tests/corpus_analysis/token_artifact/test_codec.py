from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.token_artifact.codec import decode_token_record, encode_token_record
from nlpo_toolkit.corpus_analysis.token_artifact.errors import TokenArtifactRowError

from .conftest import make_record


def test_record_row_round_trip_and_optional_encoding() -> None:
    record = make_record(section=None, char_start_in_text=None)
    row = encode_token_record(record)
    assert row["section"] == ""
    assert row["char_start_in_text"] == ""
    assert decode_token_record(row, source_path=Path("tokens.tsv"), line_number=2) == record


@pytest.mark.parametrize(("column", "value"), [
    ("chunk_index", ""), ("chunk_index", "x"), ("chunk_index", "-1"),
    ("included", "yes"), ("included", "1"), ("included", "True"),
])
def test_decode_rejects_invalid_values_with_location(column, value) -> None:
    row = encode_token_record(make_record())
    row[column] = value
    with pytest.raises(TokenArtifactRowError, match=rf"{column}.*tokens.tsv:7"):
        decode_token_record(row, source_path=Path("tokens.tsv"), line_number=7)


def test_decode_rejects_missing_column() -> None:
    row = encode_token_record(make_record())
    del row["token_index"]
    with pytest.raises(TokenArtifactRowError, match="token_index"):
        decode_token_record(row, source_path=Path("tokens.tsv"), line_number=2)
