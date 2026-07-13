from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.analysis_policy import AnalysisExtractionPolicy
from nlpo_toolkit.corpus_analysis.config import ensure_app_config
from nlpo_toolkit.corpus_analysis.dependencies import (
    AnalysisDependencies,
    CorpusPlanningDependencies,
    RunnerDependencies,
)
from nlpo_toolkit.corpus_analysis.runtime import prepare_run_context
from nlpo_toolkit.backends import BuiltNLPBackend, NLPBackendInfo
from nlpo_toolkit.corpus_analysis.run_plan import (
    AnalysisPlan,
    build_analysis_plan,
    build_count_plan,
    validate_count_plan,
)
from nlpo_toolkit.corpus_analysis.config_references import ConfigReferenceError
from nlpo_toolkit.latin.cleaners.config_loader import inspect_cleaner_config


def _write_config(path: Path) -> None:
    path.write_text("dummy", encoding="utf-8")


def _loader(data: dict, *, cleaner=None):
    config = ensure_app_config(data)

    def unexpected_cleaner_loader():
        if cleaner is None:
            raise AssertionError("cleaner loader must not be called")
        return cleaner

    return CorpusPlanningDependencies(
        load_config=lambda _path: config,
        cleaner_loader=unexpected_cleaner_loader,
    )


def _base_groups() -> dict:
    return {
        "groups": {
            "text": {"files": ["input/*.txt"]},
        },
    }


def _direct_plan(
    tmp_path: Path,
    *,
    grouping_mode: str = "groups",
    config_data: dict | None = None,
    group_files: dict | None = None,
) -> AnalysisPlan:
    config_path = tmp_path / "groups.yml"
    config_path.write_text("dummy", encoding="utf-8")
    return AnalysisPlan(
        project_root=tmp_path.resolve(),
        config_path=config_path.resolve(),
        config=ensure_app_config(
            config_data or {"groups": {"text": {"files": []}}}
        ),
        cleaned_dir=None,
        grouping_mode=grouping_mode,
        work_items=(),
        group_files=group_files or {"text": ()},
    )


@pytest.mark.parametrize(
    ("mode", "per_file", "auto_mode"),
    (
        ("groups", False, False),
        ("per_file", True, False),
        ("auto_single_cleaned", False, True),
    ),
)
def test_analysis_plan_has_one_effective_grouping_mode(
    tmp_path: Path,
    mode: str,
    per_file: bool,
    auto_mode: bool,
) -> None:
    plan = _direct_plan(tmp_path, grouping_mode=mode)
    assert plan.per_file is per_file
    assert plan.auto_mode is auto_mode


def test_analysis_plan_derives_policy_values_from_config(tmp_path: Path) -> None:
    plan = _direct_plan(
        tmp_path,
        config_data={
            "groups": {
                "whole": {"files": []},
                "part_a": {"files": []},
                "part_b": {"files": []},
            },
            "out_dir": "derived-output",
            "grouping": {"auto_group_name": "derived-group"},
            "analysis_unit": "surface",
            "validations": {
                "partitions": [
                    {
                        "name": "split",
                        "whole": "whole",
                        "parts": ["part_a", "part_b"],
                    }
                ]
            },
            "comparisons": [
                {"name": "compare", "group_a": "whole", "group_b": "part_a"}
            ],
        },
    )

    assert plan.out_dir == (tmp_path / "derived-output").resolve()
    assert plan.auto_group_name == "derived-group"
    assert [spec.name for spec in plan.partition_specs] == ["split"]
    assert [spec.name for spec in plan.comparison_specs] == ["compare"]
    assert plan.analysis_unit == "surface"
    assert plan.use_lemma is False
    assert plan.csv_header == ("word", "frequency")

    lemma_plan = _direct_plan(
        tmp_path,
        config_data={"groups": {"text": {"files": []}}},
    )
    assert lemma_plan.analysis_unit == "lemma"
    assert lemma_plan.use_lemma is True
    assert lemma_plan.csv_header == ("lemma", "count")


def test_analysis_plan_stores_only_canonical_values_and_is_frozen(
    tmp_path: Path,
) -> None:
    derived = {
        "per_file",
        "auto_mode",
        "out_dir",
        "auto_group_name",
        "partition_specs",
        "comparison_specs",
        "analysis_unit",
        "use_lemma",
        "csv_header",
    }
    assert derived.isdisjoint({field.name for field in fields(AnalysisPlan)})
    plan = _direct_plan(tmp_path)
    with pytest.raises(FrozenInstanceError):
        plan.cleaned_dir = tmp_path  # type: ignore[misc]
    with pytest.raises(TypeError):
        AnalysisPlan(
            project_root=tmp_path,
            config_path=tmp_path / "groups.yml",
            config=plan.config,
            cleaned_dir=None,
            grouping_mode="groups",
            work_items=(),
            group_files={},
            out_dir=tmp_path / "output",  # type: ignore[call-arg]
        )


def test_analysis_plan_copies_group_files_into_read_only_mapping(
    tmp_path: Path,
) -> None:
    source = {"text": (tmp_path / "a.txt",)}
    plan = _direct_plan(tmp_path, group_files=source)
    source["text"] = (tmp_path / "changed.txt",)
    assert plan.group_files["text"] == (tmp_path / "a.txt",)
    with pytest.raises(TypeError):
        plan.group_files["new"] = ()  # type: ignore[index]


def test_build_count_plan_inspects_cleaner_once(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("a", encoding="utf-8")
    cleaner_path = config_dir / "cleaner.yml"
    cleaner_path.write_text(
        "kind: scholastic_text\ninput: ../input\noutput: ../cleaned\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "groups.yml"
    _write_config(config_path)
    calls: list[Path] = []

    def inspector(path: Path):
        calls.append(path)
        return inspect_cleaner_config(path)

    dependencies = CorpusPlanningDependencies(
        load_config=lambda _path: ensure_app_config(
            {
                "preprocess": {
                    "kind": "cleaner",
                    "config": "config/cleaner.yml",
                },
                "groups": {"text": {"files": ["input/*.txt"]}},
            }
        ),
        cleaner_loader=lambda: pytest.fail("cleaner must not execute"),
        cleaner_inspector=inspector,
    )

    plan = build_count_plan(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=False,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        dependencies=dependencies,
        preprocess_mode="inspect",
    )

    assert calls == [cleaner_path.resolve()]
    assert plan.cleaner_inspection is not None
    assert plan.cleaner_inspection.input_files == ((input_dir / "a.txt").resolve(),)
    assert plan.config_files.path("root_config") == config_path.resolve()
    assert plan.config_files.path("preprocess.config") == cleaner_path.resolve()


def test_build_count_plan_has_no_output_directory_side_effect(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("a", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    _write_config(config_path)

    plan = build_count_plan(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=None,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        dependencies=_loader({**_base_groups(), "out_dir": "output"}),
        preprocess_mode="inspect",
    )

    assert plan.project_root == tmp_path.resolve()
    assert plan.config_path == config_path.resolve()
    assert plan.out_dir == (tmp_path / "output").resolve()
    assert not plan.out_dir.exists()
    assert plan.grouping_mode == "groups"
    assert plan.per_file is False
    assert plan.group_files == {"text": ((tmp_path / "input" / "a.txt").resolve(),)}
    assert plan.work_items[0].label == "text"


def test_build_analysis_plan_returns_canonical_effective_mode(
    tmp_path: Path,
) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("a", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    _write_config(config_path)

    plan = build_analysis_plan(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=True,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        dependencies=_loader(_base_groups()),
        preprocess_mode="inspect",
    )

    assert isinstance(plan, AnalysisPlan)
    assert plan.grouping_mode == "per_file"
    assert plan.per_file is True


def test_cli_auto_mode_overrides_per_file_mode(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    selected = cleaned / "only.txt"
    selected.write_text("cleaned", encoding="utf-8")
    (config_dir / "cleaner.yml").write_text(
        "kind: scholastic_text\ninput: ../cleaned\noutput: ../cleaned\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "groups.yml"
    _write_config(config_path)

    plan = build_analysis_plan(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=True,
        auto_single_cleaned=True,
        error_on_empty_group=False,
        dependencies=_loader(
            {
                "preprocess": {"kind": "cleaner", "config": "config/cleaner.yml"},
                "groups": {"text": {"files": ["{cleaned_dir}/*.txt"]}},
                "grouping": {"mode": "per_file"},
            }
        ),
        preprocess_mode="inspect",
    )

    assert plan.grouping_mode == "auto_single_cleaned"
    assert plan.auto_mode is True
    assert plan.per_file is False
    assert plan.group_files == {"text": (selected.resolve(),)}


def test_build_analysis_plan_defers_count_specific_validation(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "groups.yml"
    _write_config(config_path)
    dependencies = _loader(
        {
            "groups": {
                "whole": {"files": ["input/whole.txt"]},
                "part_a": {"files": ["input/part-a.txt"]},
                "part_b": {"files": ["input/part-b.txt"]},
            },
            "validations": {
                "partitions": [
                    {
                        "name": "split",
                        "whole": "whole",
                        "parts": ["part_a", "part_b"],
                    }
                ]
            },
        }
    )

    plan = build_analysis_plan(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=True,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        dependencies=dependencies,
        preprocess_mode="inspect",
    )

    assert plan.grouping_mode == "per_file"
    with pytest.raises(ValueError, match="validations.partitions"):
        validate_count_plan(plan)


def test_build_count_plan_returns_same_analysis_plan_object(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import nlpo_toolkit.corpus_analysis.run_plan as plan_module

    expected = _direct_plan(tmp_path)
    monkeypatch.setattr(
        plan_module,
        "build_analysis_plan",
        lambda **_kwargs: expected,
    )

    actual = plan_module.build_count_plan(
        project_root=tmp_path,
        script_dir=None,
        config_path=tmp_path / "groups.yml",
        group_by_file=False,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        dependencies=_loader(_base_groups()),
        preprocess_mode="inspect",
    )

    assert actual is expected
    assert actual.config_files is expected.config_files


def test_inspect_mode_does_not_run_cleaner(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "cleaned").mkdir()
    (tmp_path / "cleaned" / "a.txt").write_text("cleaned", encoding="utf-8")
    cleaner_config = tmp_path / "config" / "cleaner.yml"
    cleaner_config.write_text(
        "kind: scholastic_text\ninput: .\noutput: ../cleaned\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yml"
    _write_config(config_path)

    class FailingCleaner:
        @staticmethod
        def main(argv):
            raise AssertionError("cleaner must not run in inspect mode")

    plan = build_count_plan(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=None,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        dependencies=_loader(
            {
                "preprocess": {"kind": "cleaner", "config": "config/cleaner.yml"},
                "groups": {"text": {"files": ["{cleaned_dir}/*.txt"]}},
            },
            cleaner=FailingCleaner,
        ),
        preprocess_mode="inspect",
    )

    assert plan.cleaned_dir == (tmp_path / "cleaned").resolve()
    assert plan.group_files["text"] == ((tmp_path / "cleaned" / "a.txt").resolve(),)


def test_execute_mode_runs_cleaner_before_resolving_groups(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    cleaner_config = tmp_path / "config" / "cleaner.yml"
    cleaner_config.write_text(
        "kind: scholastic_text\ninput: .\noutput: ../cleaned\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yml"
    _write_config(config_path)

    class FakeCleaner:
        @staticmethod
        def main(argv):
            cleaned = Path(argv[0]).parent.parent / "cleaned"
            cleaned.mkdir()
            (cleaned / "made.txt").write_text("cleaned", encoding="utf-8")

    plan = build_count_plan(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=None,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        dependencies=_loader(
            {
                "preprocess": {"kind": "cleaner", "config": "config/cleaner.yml"},
                "groups": {"text": {"files": ["{cleaned_dir}/*.txt"]}},
            },
            cleaner=FakeCleaner,
        ),
        preprocess_mode="execute",
    )

    assert plan.group_files["text"] == ((tmp_path / "cleaned" / "made.txt").resolve(),)
    assert plan.work_items[0].files == plan.group_files["text"]


def test_inspect_and_execute_match_when_cleaned_files_already_exist(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    (cleaned / "a.txt").write_text("cleaned", encoding="utf-8")
    cleaner_config = tmp_path / "config" / "cleaner.yml"
    cleaner_config.write_text(
        "kind: scholastic_text\ninput: .\noutput: ../cleaned\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yml"
    _write_config(config_path)
    data = {
        "preprocess": {"kind": "cleaner", "config": "config/cleaner.yml"},
        "groups": {
            "full": {"files": ["{cleaned_dir}/a.txt"]},
            "part_a": {"files": ["{cleaned_dir}/a.txt"]},
            "part_b": {"files": ["{cleaned_dir}/a.txt"]},
        },
        "validations": {
            "partitions": [
                {"name": "split", "whole": "full", "parts": ["part_a", "part_b"]}
            ]
        },
        "comparisons": [
            {"name": "compare", "group_a": "part_a", "group_b": "part_b"}
        ],
    }

    class NoopCleaner:
        @staticmethod
        def main(argv):
            return None

    inspect_plan = build_count_plan(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=None,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        dependencies=_loader(data),
        preprocess_mode="inspect",
    )
    execute_plan = build_count_plan(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=None,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        dependencies=_loader(data, cleaner=NoopCleaner),
        preprocess_mode="execute",
    )

    assert inspect_plan.grouping_mode == execute_plan.grouping_mode
    assert inspect_plan.per_file == execute_plan.per_file
    assert inspect_plan.auto_mode == execute_plan.auto_mode
    assert inspect_plan.group_files == execute_plan.group_files
    assert inspect_plan.work_items == execute_plan.work_items
    assert inspect_plan.partition_specs == execute_plan.partition_specs
    assert inspect_plan.comparison_specs == execute_plan.comparison_specs


def test_yaml_per_file_mode_rejects_partition_and_comparison(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    _write_config(config_path)
    groups = {
        "full": {"files": ["input/full.txt"]},
        "part_a": {"files": ["input/a.txt"]},
        "part_b": {"files": ["input/b.txt"]},
    }

    with pytest.raises(ValueError, match="validations.partitions"):
        build_count_plan(
            project_root=tmp_path,
            script_dir=None,
            config_path=config_path,
            group_by_file=False,
            auto_single_cleaned=False,
            error_on_empty_group=False,
            dependencies=_loader(
                {
                    "groups": groups,
                    "grouping": {"mode": "per_file"},
                    "validations": {
                        "partitions": [
                            {"name": "split", "whole": "full", "parts": ["part_a", "part_b"]}
                        ]
                    },
                }
            ),
            preprocess_mode="inspect",
        )

    with pytest.raises(ValueError, match="comparisons"):
        build_count_plan(
            project_root=tmp_path,
            script_dir=None,
            config_path=config_path,
            group_by_file=False,
            auto_single_cleaned=False,
            error_on_empty_group=False,
            dependencies=_loader(
                {
                    "groups": {"a": {"files": ["input/a.txt"]}, "b": {"files": ["input/b.txt"]}},
                    "grouping": {"mode": "per_file"},
                    "comparisons": [{"name": "ab", "group_a": "a", "group_b": "b"}],
                }
            ),
            preprocess_mode="inspect",
        )


def test_cli_group_by_file_rejects_partition(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    _write_config(config_path)
    with pytest.raises(ValueError, match="validations.partitions"):
        build_count_plan(
            project_root=tmp_path,
            script_dir=None,
            config_path=config_path,
            group_by_file=True,
            auto_single_cleaned=False,
            error_on_empty_group=False,
            dependencies=_loader(
                {
                    "groups": {
                        "full": {"files": ["input/full.txt"]},
                        "part_a": {"files": ["input/a.txt"]},
                        "part_b": {"files": ["input/b.txt"]},
                    },
                    "validations": {
                        "partitions": [
                            {"name": "split", "whole": "full", "parts": ["part_a", "part_b"]}
                        ]
                    },
                }
            ),
            preprocess_mode="inspect",
        )


def test_auto_single_cleaned_selects_one_and_ignores_dotfiles(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    selected = cleaned / "only.txt"
    selected.write_text("cleaned", encoding="utf-8")
    (cleaned / ".DS_Store").write_text("ignored", encoding="utf-8")
    cleaner_config = tmp_path / "config" / "cleaner.yml"
    cleaner_config.write_text(
        "kind: scholastic_text\ninput: .\noutput: ../cleaned\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yml"
    _write_config(config_path)

    plan = build_count_plan(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=None,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        dependencies=_loader(
            {
                "groups": {"text": {"files": ["{cleaned_dir}/*.txt"]}},
                "preprocess": {"kind": "cleaner", "config": "config/cleaner.yml"},
                "grouping": {"mode": "auto_single_cleaned", "auto_group_name": "text"},
            }
        ),
        preprocess_mode="inspect",
    )

    assert plan.auto_mode is True
    assert plan.group_files == {"text": (selected.resolve(),)}


def test_auto_single_cleaned_rejects_zero_and_multiple_files(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    cleaner_config = tmp_path / "config" / "cleaner.yml"
    cleaner_config.write_text(
        "kind: scholastic_text\ninput: .\noutput: ../cleaned\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yml"
    _write_config(config_path)
    data = {
        "groups": {"text": {"files": ["{cleaned_dir}/*.txt"]}},
        "preprocess": {"kind": "cleaner", "config": "config/cleaner.yml"},
        "grouping": {"mode": "auto_single_cleaned"},
    }

    with pytest.raises(ValueError, match="no .txt files"):
        build_count_plan(
            project_root=tmp_path,
            script_dir=None,
            config_path=config_path,
            group_by_file=None,
            auto_single_cleaned=False,
            error_on_empty_group=False,
            dependencies=_loader(data),
            preprocess_mode="inspect",
        )

    (cleaned / "a.txt").write_text("a", encoding="utf-8")
    (cleaned / "b.txt").write_text("b", encoding="utf-8")
    with pytest.raises(ValueError, match="expected exactly one"):
        build_count_plan(
            project_root=tmp_path,
            script_dir=None,
            config_path=config_path,
            group_by_file=None,
            auto_single_cleaned=False,
            error_on_empty_group=False,
            dependencies=_loader(data),
            preprocess_mode="inspect",
        )


def test_empty_group_policy_and_spec_references(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "whole.txt").write_text("whole", encoding="utf-8")
    (tmp_path / "input" / "part_a.txt").write_text("part_a", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    _write_config(config_path)

    plan = build_count_plan(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=None,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        dependencies=_loader({"groups": {"empty": {"files": ["input/no_match*.txt"]}}}),
        preprocess_mode="inspect",
    )
    assert plan.group_files == {"empty": ()}

    with pytest.raises(ValueError, match="No files matched"):
        build_count_plan(
            project_root=tmp_path,
            script_dir=None,
            config_path=config_path,
            group_by_file=None,
            auto_single_cleaned=False,
            error_on_empty_group=True,
            dependencies=_loader({"groups": {"empty": {"files": ["input/no_match*.txt"]}}}),
            preprocess_mode="inspect",
        )

    with pytest.raises(ValueError, match="Partition split references empty group: part_b"):
        build_count_plan(
            project_root=tmp_path,
            script_dir=None,
            config_path=config_path,
            group_by_file=None,
            auto_single_cleaned=False,
            error_on_empty_group=False,
            dependencies=_loader(
                {
                    "groups": {
                        "whole": {"files": ["input/whole.txt"]},
                        "part_a": {"files": ["input/part_a.txt"]},
                        "part_b": {"files": ["input/missing.txt"]},
                    },
                    "validations": {
                        "partitions": [
                            {"name": "split", "whole": "whole", "parts": ["part_a", "part_b"]}
                        ]
                    },
                }
            ),
            preprocess_mode="inspect",
        )


def test_comparison_empty_reference_fails_plan(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("a", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    _write_config(config_path)

    with pytest.raises(ValueError, match="comparison ab references empty group: b"):
        build_count_plan(
            project_root=tmp_path,
            script_dir=None,
            config_path=config_path,
            group_by_file=None,
            auto_single_cleaned=False,
            error_on_empty_group=False,
            dependencies=_loader(
                {
                    "groups": {
                        "a": {"files": ["input/a.txt"]},
                        "b": {"files": ["input/missing.txt"]},
                    },
                    "comparisons": [{"name": "ab", "group_a": "a", "group_b": "b"}],
                }
            ),
            preprocess_mode="inspect",
        )


def test_prepare_run_context_validates_plan_before_nlp_initialization(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    _write_config(config_path)
    calls: list[object] = []

    def backend_factory(config):
        calls.append(config)
        return BuiltNLPBackend(
            backend=object(),
            info=NLPBackendInfo(name="fake", language="la"),
        )

    deps = RunnerDependencies(
        planning=_loader(
            {
                "groups": {
                    "a": {"files": ["input/a.txt"]},
                    "b": {"files": ["input/missing.txt"]},
                },
                "comparisons": [{"name": "ab", "group_a": "a", "group_b": "b"}],
            }
        ),
        analysis=AnalysisDependencies(
            backend_factory=backend_factory,
            extraction_policy=AnalysisExtractionPolicy(),
        ),
    )

    with pytest.raises(ValueError, match="comparison ab references empty group: a"):
        prepare_run_context(
            project_root=tmp_path,
            script_dir=None,
            config_path=config_path,
            group_by_file=None,
            auto_single_cleaned=False,
            error_on_empty_group=False,
            dependencies=deps,
        )

    assert calls == []


def test_missing_non_cleaner_reference_fails_before_cleaner_execution(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("a", encoding="utf-8")
    cleaner_path = config_dir / "cleaner.yml"
    cleaner_path.write_text(
        "kind: scholastic_text\ninput: ../input\noutput: ../cleaned\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "groups.yml"
    _write_config(config_path)
    cleaner_calls: list[object] = []

    class UnexpectedCleaner:
        @staticmethod
        def main(argv):
            cleaner_calls.append(argv)

    dependencies = _loader(
        {
            "preprocess": {"kind": "cleaner", "config": "config/cleaner.yml"},
            "groups": {"text": {"files": ["{cleaned_dir}/*.txt"]}},
            "dictcheck": {"lemma_normalize": "config/missing.tsv"},
        },
        cleaner=UnexpectedCleaner,
    )

    with pytest.raises(ConfigReferenceError, match="dictcheck.lemma_normalize"):
        build_count_plan(
            project_root=tmp_path,
            script_dir=None,
            config_path=config_path,
            group_by_file=False,
            auto_single_cleaned=False,
            error_on_empty_group=False,
            dependencies=dependencies,
            preprocess_mode="execute",
        )

    assert cleaner_calls == []
