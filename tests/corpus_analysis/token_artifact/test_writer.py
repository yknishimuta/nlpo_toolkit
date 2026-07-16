from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.token_artifact.errors import TokenArtifactWriterStateError
from nlpo_toolkit.corpus_analysis.token_artifact.paths import token_artifact_metadata_path
from nlpo_toolkit.corpus_analysis.token_artifact.schema import TokenArtifactDescriptor
from nlpo_toolkit.corpus_analysis.token_artifact.writer import TokenArtifactWriter

from .conftest import make_record


def _writer(path: Path) -> TokenArtifactWriter:
    return TokenArtifactWriter(
        path, metadata_path=token_artifact_metadata_path(path),
        descriptor=TokenArtifactDescriptor("text", ("input.txt",), "lemma"),
    )


def test_writer_publishes_body_and_complete_metadata(tmp_path: Path) -> None:
    path = tmp_path / "custom.tsv"
    writer = _writer(path)
    with writer:
        writer.write(make_record())
        writer.write(make_record(included=False, exclusion_reason="filtered"))
    assert path.exists() and token_artifact_metadata_path(path).exists()
    assert writer.final_metadata is not None
    assert writer.final_metadata.row_count == 2
    assert writer.final_metadata.included_row_count == 1
    assert writer.final_metadata.excluded_row_count == 1
    assert writer.final_metadata.sha256
    assert writer.final_metadata.size_bytes == path.stat().st_size


def test_writer_lifecycle_is_strict(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "tokens.tsv")
    with pytest.raises(TokenArtifactWriterStateError):
        writer.write(make_record())
    with writer:
        with pytest.raises(TokenArtifactWriterStateError):
            writer.__enter__()
    with pytest.raises(TokenArtifactWriterStateError):
        writer.write(make_record())
    with pytest.raises(TokenArtifactWriterStateError):
        writer.__enter__()


def test_body_failure_publishes_nothing_and_cleans_unique_temps(tmp_path: Path) -> None:
    path = tmp_path / "tokens.tsv"
    with pytest.raises(RuntimeError):
        with _writer(path) as writer:
            writer.write(make_record())
            raise RuntimeError("boom")
    assert not path.exists()
    assert not token_artifact_metadata_path(path).exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_writer_rejects_nonprotocol_metadata_path_before_creation(tmp_path: Path) -> None:
    with pytest.raises(TokenArtifactWriterStateError, match="protocol"):
        TokenArtifactWriter(
            tmp_path / "tokens.tsv", metadata_path=tmp_path / "other.json",
            descriptor=TokenArtifactDescriptor("g"),
        )
    assert list(tmp_path.iterdir()) == []
