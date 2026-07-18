import json
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.token_artifact.errors import (
    TokenArtifactMetadataError,
)
from nlpo_toolkit.corpus_analysis.token_artifact.schema import (
    TOKEN_ARTIFACT_COLUMNS,
    TOKEN_ARTIFACT_SCHEMA_NAME,
    TOKEN_ARTIFACT_SCHEMA_VERSION,
    TokenArtifactDescriptor,
    TokenArtifactFilterDescriptor,
    TokenArtifactMetadata,
    TokenArtifactNLPDescriptor,
    metadata_from_json,
    metadata_to_json,
)


def _metadata(**changes) -> TokenArtifactMetadata:
    values = dict(
        complete=True,
        row_count=1,
        included_row_count=1,
        excluded_row_count=0,
        group="g",
        source_files=("a.txt",),
        analysis_unit="lemma",
        upos_targets=("NOUN",),
        nlp=TokenArtifactNLPDescriptor(backend="fake"),
        filters=TokenArtifactFilterDescriptor(),
        artifact_path="/old/tokens.tsv",
        sha256="a" * 64,
        size_bytes=10,
    )
    values.update(changes)
    return TokenArtifactMetadata(**values)


def test_descriptor_freezes_copies_and_writer_schema_is_version_two() -> None:
    nlp = TokenArtifactNLPDescriptor(backend="fake")
    descriptor = TokenArtifactDescriptor(
        "g",
        ["a.txt"],
        "lemma",
        ["NOUN"],
        nlp,
        TokenArtifactFilterDescriptor(),
    )  # type: ignore[arg-type]
    assert descriptor.source_files == ("a.txt",)
    assert descriptor.upos_targets == ("NOUN",)
    assert descriptor.nlp.backend == "fake"
    assert TOKEN_ARTIFACT_SCHEMA_NAME == "nlpo-token-artifact"
    assert TOKEN_ARTIFACT_SCHEMA_VERSION == 2
    assert TOKEN_ARTIFACT_COLUMNS[0] == "group"
    assert TOKEN_ARTIFACT_COLUMNS[-1] == "ref_tag"


def test_metadata_json_round_trip_is_deterministic() -> None:
    metadata = _metadata()
    text = metadata_to_json(metadata)
    assert text.endswith("\n")
    assert text == metadata_to_json(metadata)
    assert json.loads(text)["source_files"] == ["a.txt"]
    assert metadata_from_json(text, source_path=Path("meta.json")) == metadata


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("complete", "false"),
        ("row_count", "1"),
        ("size_bytes", 1.5),
        ("source_files", "a.txt"),
        ("nlp", []),
        ("row_count", True),
        ("row_count", -1),
        ("schema", "wrong"),
        ("schema_version", 3),
    ],
)
def test_metadata_rejects_wrong_types_and_values(field, value) -> None:
    data = _metadata().model_dump(mode="json", by_alias=True)
    data[field] = value
    with pytest.raises(TokenArtifactMetadataError):
        metadata_from_json(json.dumps(data), source_path=Path("meta.json"))


def test_metadata_rejects_unknown_keys_and_inconsistent_counts() -> None:
    data = _metadata().model_dump(mode="json", by_alias=True)
    data["unknown"] = 1
    with pytest.raises(TokenArtifactMetadataError):
        metadata_from_json(json.dumps(data), source_path=Path("meta.json"))
    with pytest.raises(ValueError, match="must equal"):
        _metadata(row_count=2)
