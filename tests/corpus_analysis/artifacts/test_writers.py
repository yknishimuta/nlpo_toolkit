from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.artifacts.models import ArtifactKind, PlannedArtifact
from nlpo_toolkit.corpus_analysis.artifacts.publication import ArtifactPublicationError
from nlpo_toolkit.corpus_analysis.artifacts.writers.frequency import write_frequency_artifact


def test_frequency_writer_uses_nonstandard_planned_path(tmp_path: Path) -> None:
    path = tmp_path / "custom-name.data"
    artifact = PlannedArtifact(ArtifactKind.FREQUENCY, path, group="latin")
    write_frequency_artifact(
        artifact,
        counter=Counter({"b": 1, "a": 2, "c": 2}),
        header=("lemma", "count"),
    )
    assert path.read_text(encoding="utf-8").splitlines() == [
        "lemma,count", "a,2", "c,2", "b,1"
    ]


def test_frequency_writer_rejects_wrong_kind_with_context(tmp_path: Path) -> None:
    artifact = PlannedArtifact(ArtifactKind.SUMMARY, tmp_path / "custom", group=None)
    with pytest.raises(ArtifactPublicationError) as error:
        write_frequency_artifact(
            artifact, counter=Counter(), header=("lemma", "count")
        )
    message = str(error.value)
    assert "expected=frequency" in message
    assert "actual=summary" in message
    assert str(artifact.path) in message
