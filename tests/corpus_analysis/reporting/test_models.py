from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path

from nlpo_toolkit.corpus_analysis.artifacts.models import ArtifactKind
from nlpo_toolkit.corpus_analysis.reporting.models import GeneratedArtifactReport


def test_generated_artifact_report_keeps_typed_values(tmp_path: Path) -> None:
    report = GeneratedArtifactReport(
        ArtifactKind.FREQUENCY, (tmp_path / "frequency").resolve(), group="g"
    )
    assert is_dataclass(report)
    assert report.kind is ArtifactKind.FREQUENCY
    assert isinstance(report.path, Path)
    assert {field.name for field in fields(report)} == {"kind", "path", "group", "name"}
