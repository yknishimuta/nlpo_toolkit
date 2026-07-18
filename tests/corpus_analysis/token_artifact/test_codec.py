from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.token_artifact.codec import (
    decode_token_record,
    encode_token_record,
)
from nlpo_toolkit.corpus_analysis.token_artifact.errors import TokenArtifactRowError

from .conftest import make_record
from nlpo_toolkit.nlp.contracts import UDMorphFeature


def test_record_row_round_trip_and_optional_encoding() -> None:
    record = make_record(section=None, char_start_in_text=None)
    row = encode_token_record(record)
    assert row["section"] == ""
    assert row["char_start_in_text"] == ""
    assert (
        decode_token_record(row, source_path=Path("tokens.tsv"), line_number=2)
        == record
    )


def test_version_two_morphology_round_trip_and_version_one_defaults_empty() -> None:
    record = make_record(morphology=(UDMorphFeature("Case", "Nom"),))
    row = encode_token_record(record)
    assert row["feats"] == "Case=Nom"
    assert (
        decode_token_record(
            row, source_path=Path("tokens.tsv"), line_number=2, schema_version=2
        )
        == record
    )
    version_one = decode_token_record(
        row, source_path=Path("tokens.tsv"), line_number=2, schema_version=1
    )
    assert version_one.morphology == ()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("chunk_index", ""),
        ("chunk_index", "x"),
        ("chunk_index", "-1"),
        ("included", "yes"),
        ("included", "1"),
        ("included", "True"),
    ],
)
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
