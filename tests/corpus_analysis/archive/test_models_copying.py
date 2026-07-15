from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.archive.copying import copy_archive_files
from nlpo_toolkit.corpus_analysis.archive.file_metadata import (
    file_sha256, read_source_file_metadata,
)
from nlpo_toolkit.corpus_analysis.archive.models import ArchiveCopySource


def test_copy_source_rejects_unsafe_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="safe relative"):
        ArchiveCopySource(source.resolve(), Path("../escape.txt"))


def test_copying_preserves_content_metadata_and_resolves_collisions(tmp_path: Path) -> None:
    first = tmp_path / "a/same.txt"
    second = tmp_path / "b/same.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    archive = tmp_path / "archive"
    archive.mkdir()
    copied = copy_archive_files(
        (
            ArchiveCopySource(first.resolve(), Path("same.txt")),
            ArchiveCopySource(second.resolve(), Path("same.txt")),
        ),
        destination_root=archive / "outputs",
        archive_directory=archive,
    )
    assert [item.archive_relative_path for item in copied] == [
        Path("outputs/same.txt"), Path("outputs/same_2.txt")
    ]
    assert (archive / copied[0].archive_relative_path).read_text() == "first"
    assert copied[0].sha256 == file_sha256(first)


def test_file_metadata_handles_empty_and_large_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    large = tmp_path / "large"
    content = b"a" * (1024 * 1024 + 3)
    large.write_bytes(content)
    assert read_source_file_metadata(empty).sha256 == hashlib.sha256(b"").hexdigest()
    metadata = read_source_file_metadata(large)
    assert metadata.path == large.resolve()
    assert metadata.size_bytes == len(content)
    assert metadata.sha256 == hashlib.sha256(content).hexdigest()
