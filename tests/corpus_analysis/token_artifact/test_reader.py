import json
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.token_artifact.errors import TokenArtifactError
from nlpo_toolkit.corpus_analysis.token_artifact.paths import token_artifact_metadata_path
from nlpo_toolkit.corpus_analysis.token_artifact.reader import (
    read_token_artifact_metadata,
    read_token_records,
)
from nlpo_toolkit.corpus_analysis.token_artifact.schema import TokenArtifactDescriptor
from nlpo_toolkit.corpus_analysis.token_artifact.writer import TokenArtifactWriter

from .conftest import make_record


def _write(path: Path):
    with TokenArtifactWriter(
        path, metadata_path=token_artifact_metadata_path(path),
        descriptor=TokenArtifactDescriptor("g"),
    ) as writer:
        writer.write(make_record())


def test_reader_reads_valid_artifact_and_ignores_metadata_original_location(tmp_path: Path) -> None:
    original = tmp_path / "original" / "tokens.tsv"
    original.parent.mkdir()
    _write(original)
    archive = tmp_path / "archive"
    archive.mkdir()
    moved = archive / "tokens.tsv"
    moved.write_bytes(original.read_bytes())
    token_artifact_metadata_path(moved).write_bytes(
        token_artifact_metadata_path(original).read_bytes()
    )
    assert list(read_token_records(moved, verify_hash=True)) == [make_record()]
    assert read_token_artifact_metadata(moved).artifact_path == str(original.resolve())


def test_reader_rejects_missing_and_directory_paths(tmp_path: Path) -> None:
    with pytest.raises(TokenArtifactError, match="not found"):
        list(read_token_records(tmp_path / "missing.tsv"))
    directory = tmp_path / "tokens.tsv"
    directory.mkdir()
    with pytest.raises(TokenArtifactError, match="regular file"):
        list(read_token_records(directory))


@pytest.mark.parametrize("content", [b"\xff", b"not json", b"[]"])
def test_metadata_rejects_invalid_utf8_json_and_root(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "tokens.tsv"
    path.write_text("", encoding="utf-8")
    token_artifact_metadata_path(path).write_bytes(content)
    with pytest.raises(TokenArtifactError):
        read_token_artifact_metadata(path)


def test_reader_rejects_header_row_counts_size_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "tokens.tsv"
    _write(path)
    path.write_text("bad\theader\n", encoding="utf-8")
    with pytest.raises(TokenArtifactError, match="header"):
        list(read_token_records(path))

    _write(path)
    meta_path = token_artifact_metadata_path(path)
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    data["row_count"] = 2
    data["included_row_count"] = 2
    meta_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(TokenArtifactError, match="count mismatch"):
        list(read_token_records(path))

    _write(path)
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    data["size_bytes"] += 1
    meta_path.write_text(json.dumps(data), encoding="utf-8")
    assert len(list(read_token_records(path, verify_hash=False))) == 1
    with pytest.raises(TokenArtifactError, match="size mismatch"):
        list(read_token_records(path, verify_hash=True))


def test_reader_rejects_incomplete_before_tsv_iteration(tmp_path: Path) -> None:
    path = tmp_path / "tokens.tsv"
    _write(path)
    meta_path = token_artifact_metadata_path(path)
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    data["complete"] = False
    data["sha256"] = ""
    meta_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(TokenArtifactError, match="incomplete"):
        list(read_token_records(path))
