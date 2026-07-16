from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.artifacts.models import (
    ArtifactKind, ArtifactPlan, PlannedArtifact,
)
from nlpo_toolkit.corpus_analysis.runtime import prepare_run_context
from tests.corpus_analysis.fake_nlp import (
    corpus_request, fake_backend_factory, runner_dependencies,
)


def artifact(kind: ArtifactKind, path: Path, *, group: str | None = None,
             name: str | None = None) -> PlannedArtifact:
    return PlannedArtifact(kind, path, group=group, name=name)


def _prepare(tmp_path: Path, config: dict) -> ArtifactPlan:
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy", encoding="utf-8")
    return prepare_run_context(
        corpus_request(tmp_path, config_path),
        dependencies=runner_dependencies(
            lambda _path: config, fake_backend_factory([])
        ),
    ).artifact_plan


def test_minimal_count_plan_has_exact_artifacts(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "text.txt").write_text("text", encoding="utf-8")
    plan = _prepare(tmp_path, {
        "out_dir": "output",
        "groups": {"text": {"files": ["input/text.txt"]}},
    })
    assert [(item.kind, item.group, item.name, item.path.name)
            for item in plan.artifacts] == [
        (ArtifactKind.FREQUENCY, "text", None, "frequency_text.csv"),
        (ArtifactKind.SUMMARY, None, None, "summary.txt"),
        (ArtifactKind.RUN_METADATA, None, None, "run_meta.json"),
    ]


def test_all_enabled_count_plan_is_complete_and_deterministic(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    for name in ("full", "part", "part2"):
        (tmp_path / "input" / f"{name}.txt").write_text(name, encoding="utf-8")
    (tmp_path / "words.txt").write_text("full\n", encoding="utf-8")
    (tmp_path / "tags.txt").write_text("TAG\n", encoding="utf-8")
    config = {
        "out_dir": "output",
        "groups": {
            "full": {"files": ["input/full.txt"]},
            "part": {"files": ["input/part.txt"]},
            "part2": {"files": ["input/part2.txt"]},
        },
        "dictcheck": {"enabled": True, "wordlist": "words.txt"},
        "ref_tags": {"enabled": True, "patterns": "tags.txt"},
        "trace": {"enabled": True, "path": "output/trace.tsv"},
        "artifacts": {"tokens": {"enabled": True, "path": "output/tokens.tsv"}},
        "validations": {"partitions": [{
            "name": "split", "whole": "full", "parts": ["part", "part2"],
        }]},
        "comparisons": [{
            "name": "full-v-part", "group_a": "full", "group_b": "part",
        }],
    }
    first = _prepare(tmp_path, config)
    second = _prepare(tmp_path, config)
    assert first.paths == second.paths
    kinds = [item.kind for item in first.artifacts]
    for kind in ArtifactKind:
        assert kind in kinds
    for group in ("full", "part", "part2"):
        assert len(first.select(group=group)) == 7
    assert first.require(ArtifactKind.PARTITION_VALIDATION_CSV, name="split")
    assert first.require(ArtifactKind.GROUP_COMPARISON_CSV, name="full-v-part")


def test_plan_normalizes_paths_and_supports_typed_queries(tmp_path: Path) -> None:
    plan = ArtifactPlan((
        artifact(ArtifactKind.FREQUENCY, tmp_path / "frequency.csv", group="g"),
        artifact(ArtifactKind.SUMMARY, tmp_path / "summary.txt"),
        artifact(ArtifactKind.RUN_METADATA, tmp_path / "run_meta.json"),
    ))
    assert plan.require(ArtifactKind.FREQUENCY, group="g").path.is_absolute()
    assert plan.optional(ArtifactKind.DIAGNOSTIC_TRACE, group="g") is None
    assert plan.paths == tuple(item.path for item in plan.artifacts)
    assert {field.name for field in fields(plan)} == {"artifacts"}


def test_duplicate_logical_identifier_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duplicate artifact logical identifier"):
        ArtifactPlan((
            artifact(ArtifactKind.FREQUENCY, tmp_path / "a.csv", group="g"),
            artifact(ArtifactKind.FREQUENCY, tmp_path / "b.csv", group="g"),
        ))


@pytest.mark.parametrize(("first_kind", "second_kind", "first_owner", "second_owner"), (
    (ArtifactKind.TOKEN_ARTIFACT, ArtifactKind.FREQUENCY, "tokens", "frequency"),
    (ArtifactKind.TOKEN_ARTIFACT_METADATA, ArtifactKind.SUMMARY, "metadata", "summary"),
    (ArtifactKind.DIAGNOSTIC_TRACE, ArtifactKind.DICTCHECK_KNOWN, "trace", "known"),
))
def test_path_collision_reports_both_artifacts(
    tmp_path: Path, first_kind: ArtifactKind, second_kind: ArtifactKind,
    first_owner: str, second_owner: str,
) -> None:
    path = (tmp_path / "same.file").resolve()
    first_group = "g" if first_kind not in {ArtifactKind.SUMMARY} else None
    second_group = "g" if second_kind not in {ArtifactKind.SUMMARY} else None
    with pytest.raises(ValueError) as caught:
        ArtifactPlan((artifact(first_kind, path, group=first_group),
                      artifact(second_kind, path, group=second_group)))
    message = str(caught.value)
    assert first_kind.value in message and second_kind.value in message
    assert str(path) in message


def test_case_only_collision_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Artifact path collision"):
        ArtifactPlan((
            artifact(ArtifactKind.FREQUENCY, tmp_path / "A.csv", group="A"),
            artifact(ArtifactKind.FREQUENCY, tmp_path / "a.csv", group="a"),
        ))


def test_existing_directory_and_file_ancestor_are_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="existing directory"):
        ArtifactPlan((artifact(ArtifactKind.SUMMARY, directory),))
    ancestor = tmp_path / "file"
    ancestor.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="ancestor"):
        ArtifactPlan((artifact(ArtifactKind.SUMMARY, ancestor / "summary.txt"),))


def test_collision_precedes_backend_and_output_creation(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "text.txt").write_text("text", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy", encoding="utf-8")
    backend_calls = 0

    def backend_factory(_config):
        nonlocal backend_calls
        backend_calls += 1
        raise AssertionError("backend must not be constructed")

    config = {
        "out_dir": "output",
        "groups": {"text": {"files": ["input/text.txt"]}},
        "artifacts": {"tokens": {
            "enabled": True, "path": "output/frequency_text.csv",
        }},
    }
    with pytest.raises(ValueError, match="Artifact path collision"):
        prepare_run_context(
            corpus_request(tmp_path, config_path),
            dependencies=runner_dependencies(lambda _path: config, backend_factory),
        )
    assert backend_calls == 0
    assert not (tmp_path / "output").exists()
