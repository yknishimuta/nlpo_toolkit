from pathlib import Path

import pytest

import nlpo_toolkit.corpus_analysis.planning.build as build_module
from nlpo_toolkit.corpus_analysis.config import ensure_app_config
from nlpo_toolkit.corpus_analysis.ports import CorpusPlanningDependencies
from nlpo_toolkit.corpus_analysis.requests import CorpusPreparationRequest


def _request(tmp_path: Path, override=None) -> CorpusPreparationRequest:
    path = tmp_path / "groups.yml"
    path.write_text("groups: {}\n", encoding="utf-8")
    return CorpusPreparationRequest(tmp_path, path, grouping_override=override)


def test_build_reads_config_once_resolves_paths_mode_and_header(tmp_path: Path) -> None:
    calls = []
    config = ensure_app_config({
        "groups": {"g": {"files": ["input.txt"]}},
        "out_dir": "generated",
        "analysis_unit": "surface",
        "csv_header": ["token", "n"],
    })
    plan = build_module.build_analysis_plan(
        _request(tmp_path, "per_file"),
        dependencies=CorpusPlanningDependencies(
            load_config=lambda path: calls.append(path) or config,
            cleaner_inspector=lambda path: pytest.fail("inspector called"),
        ),
    )
    assert len(calls) == 1
    assert plan.grouping_mode == "per_file"
    assert plan.out_dir == (tmp_path / "generated").resolve()
    assert plan.analysis_mode.csv_header == ("token", "n")
    assert not plan.out_dir.exists()


def test_build_inspects_cleaner_once_without_executing_it(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("text", encoding="utf-8")
    cleaner_config = tmp_path / "cleaner.yml"
    cleaner_config.write_text(
        "kind: scholastic_text\ninput: input.txt\noutput: cleaned\n",
        encoding="utf-8",
    )
    config = ensure_app_config({
        "groups": {"g": {"files": ["{cleaned_dir}/*.txt"]}},
        "preprocess": {"kind": "cleaner", "config": "cleaner.yml"},
    })
    from nlpo_toolkit.latin.cleaners.config_loader import inspect_cleaner_config
    calls = []
    plan = build_module.build_analysis_plan(
        _request(tmp_path),
        dependencies=CorpusPlanningDependencies(
            load_config=lambda path: config,
            cleaner_inspector=lambda path: calls.append(path) or inspect_cleaner_config(path),
        ),
    )
    assert calls == [cleaner_config.resolve()]
    assert plan.cleaner_plan is not None
    assert plan.cleaner_plan.output_path == (tmp_path / "cleaned").resolve()
    assert not (tmp_path / "cleaned").exists()


def test_empty_groups_and_count_per_file_constraints_are_rejected(tmp_path: Path) -> None:
    dependencies = CorpusPlanningDependencies(
        load_config=lambda path: ensure_app_config({"groups": {}}),
        cleaner_inspector=lambda path: pytest.fail("inspector called"),
    )
    with pytest.raises(ValueError, match="non-empty"):
        build_module.build_analysis_plan(_request(tmp_path), dependencies=dependencies)
