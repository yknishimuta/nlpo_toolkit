from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
from nlpo_toolkit.cleaner_contracts import CleanerExecutionResult

from nlpo_toolkit.corpus_analysis.config import ensure_app_config
from nlpo_toolkit.corpus_analysis.corpus_errors import CorpusPreparationError
from nlpo_toolkit.corpus_analysis.ports import (
    CorpusPlanningDependencies,
    CorpusPreparationDependencies,
)
from nlpo_toolkit.corpus_analysis.planning.resolve import (
    inspect_analysis_plan,
    prepare_analysis_plan,
)
from nlpo_toolkit.corpus_analysis.requests import CorpusPreparationRequest
from nlpo_toolkit.corpus_analysis.planning.build import (
    build_analysis_plan,
    build_count_plan,
)
from nlpo_toolkit.corpus_analysis.planning.resolve import (
    prepare_count_plan,
)
from nlpo_toolkit.corpus_analysis.planning.validate import AnalysisPlanError
from nlpo_toolkit.latin.cleaners.config_loader import inspect_cleaner_config


def _request(tmp_path: Path, *, override: str | None = None) -> CorpusPreparationRequest:
    config_path = tmp_path / "groups.yml"
    config_path.touch()
    return CorpusPreparationRequest(
        project_root=tmp_path,
        config_path=config_path,
        grouping_override=override,  # type: ignore[arg-type]
    )


def _planning(config: dict) -> CorpusPlanningDependencies:
    return CorpusPlanningDependencies(
        load_config=lambda _path: ensure_app_config(config),
        cleaner_inspector=inspect_cleaner_config,
    )


def _cleaner_config(tmp_path: Path) -> Path:
    source = tmp_path / "source.txt"
    source.write_text("input", encoding="utf-8")
    path = tmp_path / "cleaner.yml"
    path.write_text(
        "kind: scholastic_text\ninput: source.txt\noutput: cleaned\n",
        encoding="utf-8",
    )
    return path


def _config(*, preprocess: dict | None = None, mode: str = "groups") -> dict:
    result: dict = {
        "groups": {"text": {"files": ["{cleaned_dir}/*.txt"]}},
        "grouping": {"mode": mode},
        "out_dir": "output",
    }
    if preprocess is not None:
        result["preprocess"] = preprocess
    return result


def _result(request) -> CleanerExecutionResult:
    config = request.inspection.config
    return CleanerExecutionResult(
        config.source_path, config.kind, config.output_path, (), config.ref_tsv_path
    )


def test_analysis_plan_is_static_and_planning_has_no_loader(tmp_path: Path) -> None:
    _cleaner_config(tmp_path)
    plan = build_analysis_plan(
        _request(tmp_path),
        dependencies=_planning(
            _config(preprocess={"kind": "cleaner", "config": "cleaner.yml"})
        ),
    )

    assert {field.name for field in fields(type(plan))}.isdisjoint(
        {"cleaned_dir", "work_items", "group_files"}
    )
    assert {field.name for field in fields(CorpusPlanningDependencies)} == {
        "load_config",
        "cleaner_inspector",
    }
    assert not (tmp_path / "cleaned").exists()
    assert not (tmp_path / "output").exists()


def test_inspection_resolves_current_files_without_running_cleaner(tmp_path: Path) -> None:
    _cleaner_config(tmp_path)
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    existing = cleaned / "existing.txt"
    existing.write_text("old", encoding="utf-8")
    plan = build_analysis_plan(
        _request(tmp_path),
        dependencies=_planning(
            _config(preprocess={"kind": "cleaner", "config": "cleaner.yml"})
        ),
    )

    resolved = inspect_analysis_plan(plan)

    assert resolved.group_files["text"] == (existing.resolve(),)


def test_prepare_runs_cleaner_once_then_resolves_generated_glob(tmp_path: Path) -> None:
    _cleaner_config(tmp_path)
    calls = 0

    def cleaner(request):
        nonlocal calls
        calls += 1
        output = tmp_path / "cleaned"
        output.mkdir(exist_ok=True)
        (output / "generated.txt").write_text("new", encoding="utf-8")
        return _result(request)

    definition = build_analysis_plan(
        _request(tmp_path),
        dependencies=_planning(
            _config(preprocess={"kind": "cleaner", "config": "cleaner.yml"})
        ),
    )
    resolved = prepare_analysis_plan(
        definition,
        dependencies=CorpusPreparationDependencies(execute_cleaner=cleaner),
    )

    assert calls == 1
    assert resolved.group_files["text"] == (
        (tmp_path / "cleaned" / "generated.txt").resolve(),
    )


def test_auto_single_cleaned_uses_post_cleaner_state(tmp_path: Path) -> None:
    _cleaner_config(tmp_path)

    def cleaner(request):
        output = tmp_path / "cleaned"
        output.mkdir(exist_ok=True)
        (output / "only.txt").write_text("new", encoding="utf-8")
        return _result(request)

    definition = build_analysis_plan(
        _request(tmp_path, override="auto_single_cleaned"),
        dependencies=_planning(
            _config(preprocess={"kind": "cleaner", "config": "cleaner.yml"})
        ),
    )
    resolved = prepare_analysis_plan(
        definition,
        dependencies=CorpusPreparationDependencies(execute_cleaner=cleaner),
    )

    assert resolved.work_items[0].files == (
        (tmp_path / "cleaned" / "only.txt").resolve(),
    )


def test_auto_single_cleaned_rejects_multiple_post_cleaner_files(tmp_path: Path) -> None:
    _cleaner_config(tmp_path)

    def cleaner(request):
        output = tmp_path / "cleaned"
        output.mkdir(exist_ok=True)
        for name in ("a.txt", "b.txt"):
            (output / name).write_text(name, encoding="utf-8")
        return _result(request)

    definition = build_analysis_plan(
        _request(tmp_path, override="auto_single_cleaned"),
        dependencies=_planning(
            _config(preprocess={"kind": "cleaner", "config": "cleaner.yml"})
        ),
    )
    with pytest.raises(CorpusPreparationError, match="exactly one"):
        prepare_analysis_plan(
            definition,
            dependencies=CorpusPreparationDependencies(execute_cleaner=cleaner),
        )


def test_no_cleaner_resolves_normal_groups_with_no_cleaned_dir(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("text", encoding="utf-8")
    definition = build_analysis_plan(
        _request(tmp_path),
        dependencies=_planning({"groups": {"text": {"files": ["input.txt"]}}}),
    )
    resolved = prepare_analysis_plan(
        definition,
        dependencies=CorpusPreparationDependencies(
            execute_cleaner=lambda _request: pytest.fail("service called")
        ),
    )

    assert resolved.cleaned_dir is None
    assert resolved.group_files == {"text": (source.resolve(),)}


def test_count_validation_is_split_between_structure_and_references(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("text", encoding="utf-8")
    structural = {
        "groups": {"text": {"files": ["input.txt"]}},
        "grouping": {"mode": "per_file"},
        "validations": {
            "partitions": [{"name": "p", "whole": "text", "parts": ["missing", "other"]}]
        },
    }
    with pytest.raises((AnalysisPlanError, ValueError), match="partitions"):
        build_count_plan(_request(tmp_path), dependencies=_planning(structural))

    references = {
        "groups": {"text": {"files": ["input.txt"]}},
        "validations": {
            "partitions": [{"name": "p", "whole": "text", "parts": ["missing", "other"]}]
        },
    }
    definition = build_count_plan(
        _request(tmp_path), dependencies=_planning(references)
    )
    with pytest.raises(AnalysisPlanError, match="empty group: missing"):
        prepare_count_plan(
            definition,
            dependencies=CorpusPreparationDependencies(
                execute_cleaner=lambda _request: pytest.fail("service called")
            ),
        )
