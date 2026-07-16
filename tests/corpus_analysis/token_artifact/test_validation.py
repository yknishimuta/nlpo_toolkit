from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.token_artifact.errors import TokenArtifactError
from nlpo_toolkit.corpus_analysis.token_artifact.paths import token_artifact_metadata_path
from nlpo_toolkit.corpus_analysis.token_artifact.schema import TokenArtifactDescriptor
from nlpo_toolkit.corpus_analysis.token_artifact.validation import validate_token_artifact
from nlpo_toolkit.corpus_analysis.token_artifact.writer import TokenArtifactWriter

from .conftest import make_record


def test_validation_fully_reads_and_returns_metadata(tmp_path: Path) -> None:
    path = tmp_path / "tokens.tsv"
    with TokenArtifactWriter(
        path, metadata_path=token_artifact_metadata_path(path),
        descriptor=TokenArtifactDescriptor("g"),
    ) as writer:
        writer.write(make_record())
    metadata = validate_token_artifact(path)
    assert metadata.row_count == 1


def test_validation_checks_hash_by_default(tmp_path: Path) -> None:
    path = tmp_path / "tokens.tsv"
    with TokenArtifactWriter(
        path, metadata_path=token_artifact_metadata_path(path),
        descriptor=TokenArtifactDescriptor("g"),
    ) as writer:
        writer.write(make_record())
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(TokenArtifactError):
        validate_token_artifact(path)
