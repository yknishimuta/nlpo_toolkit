from __future__ import annotations

from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.artifacts.models import ArtifactKind, PlannedArtifact
from nlpo_toolkit.corpus_analysis.artifacts.publication import ArtifactPublicationError, publish_text


def test_publish_text_replaces_final_and_cleans_temporary_file(tmp_path: Path) -> None:
    artifact = PlannedArtifact(ArtifactKind.SUMMARY, tmp_path / "odd-summary")
    publish_text(artifact, content="complete\n")
    assert artifact.path.read_text(encoding="utf-8") == "complete\n"
    assert list(tmp_path.glob("*.tmp")) == []


def test_publication_failure_does_not_replace_existing_file(tmp_path: Path) -> None:
    final = tmp_path / "summary"
    final.write_text("old", encoding="utf-8")
    artifact = PlannedArtifact(ArtifactKind.SUMMARY, final)
    with pytest.raises(ArtifactPublicationError):
        publish_text(artifact, content=None)  # type: ignore[arg-type]
    assert final.read_text(encoding="utf-8") == "old"
