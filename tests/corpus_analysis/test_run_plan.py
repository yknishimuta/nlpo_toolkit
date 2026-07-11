from __future__ import annotations

from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.runner_types import RunnerDependencies
from nlpo_toolkit.corpus_analysis.runtime import prepare_run_context
from nlpo_toolkit.backends import BuiltNLPBackend, NLPBackendInfo
from nlpo_toolkit.corpus_analysis.run_plan import build_run_plan


def _write_config(path: Path) -> None:
    path.write_text("dummy", encoding="utf-8")


def _loader(data: dict):
    return lambda _path: data


def _base_groups() -> dict:
    return {
        "groups": {
            "text": {"files": ["input/*.txt"]},
        },
    }


def test_build_run_plan_has_no_output_directory_side_effect(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("a", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    _write_config(config_path)

    plan = build_run_plan(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=None,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        load_config_fn=_loader({**_base_groups(), "out_dir": "output"}),
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


def test_inspect_mode_does_not_run_cleaner(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "cleaned").mkdir()
    (tmp_path / "cleaned" / "a.txt").write_text("cleaned", encoding="utf-8")
    cleaner_config = tmp_path / "config" / "cleaner.yml"
    cleaner_config.write_text("output: ../cleaned\n", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    _write_config(config_path)

    class FailingCleaner:
        @staticmethod
        def main(argv):
            raise AssertionError("cleaner must not run in inspect mode")

    plan = build_run_plan(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=None,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        load_config_fn=_loader(
            {
                "preprocess": {"kind": "cleaner", "config": "config/cleaner.yml"},
                "groups": {"text": {"files": ["{cleaned_dir}/*.txt"]}},
            }
        ),
        preprocess_mode="inspect",
        cleaner=FailingCleaner,
    )

    assert plan.cleaned_dir == (tmp_path / "cleaned").resolve()
    assert plan.group_files["text"] == ((tmp_path / "cleaned" / "a.txt").resolve(),)


def test_execute_mode_runs_cleaner_before_resolving_groups(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    cleaner_config = tmp_path / "config" / "cleaner.yml"
    cleaner_config.write_text("output: ../cleaned\n", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    _write_config(config_path)

    class FakeCleaner:
        @staticmethod
        def main(argv):
            cleaned = Path(argv[0]).parent.parent / "cleaned"
            cleaned.mkdir()
            (cleaned / "made.txt").write_text("cleaned", encoding="utf-8")

    plan = build_run_plan(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=None,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        load_config_fn=_loader(
            {
                "preprocess": {"kind": "cleaner", "config": "config/cleaner.yml"},
                "groups": {"text": {"files": ["{cleaned_dir}/*.txt"]}},
            }
        ),
        preprocess_mode="execute",
        cleaner=FakeCleaner,
    )

    assert plan.group_files["text"] == ((tmp_path / "cleaned" / "made.txt").resolve(),)
    assert plan.work_items[0].files == plan.group_files["text"]


def test_inspect_and_execute_match_when_cleaned_files_already_exist(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    (cleaned / "a.txt").write_text("cleaned", encoding="utf-8")
    cleaner_config = tmp_path / "config" / "cleaner.yml"
    cleaner_config.write_text("output: ../cleaned\n", encoding="utf-8")
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

    inspect_plan = build_run_plan(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=None,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        load_config_fn=_loader(data),
        preprocess_mode="inspect",
    )
    execute_plan = build_run_plan(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=None,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        load_config_fn=_loader(data),
        preprocess_mode="execute",
        cleaner=NoopCleaner,
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
        build_run_plan(
            project_root=tmp_path,
            script_dir=None,
            config_path=config_path,
            group_by_file=False,
            auto_single_cleaned=False,
            error_on_empty_group=False,
            load_config_fn=_loader(
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
        build_run_plan(
            project_root=tmp_path,
            script_dir=None,
            config_path=config_path,
            group_by_file=False,
            auto_single_cleaned=False,
            error_on_empty_group=False,
            load_config_fn=_loader(
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
        build_run_plan(
            project_root=tmp_path,
            script_dir=None,
            config_path=config_path,
            group_by_file=True,
            auto_single_cleaned=False,
            error_on_empty_group=False,
            load_config_fn=_loader(
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
    cleaner_config.write_text("output: ../cleaned\n", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    _write_config(config_path)

    plan = build_run_plan(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=None,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        load_config_fn=_loader(
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
    cleaner_config.write_text("output: ../cleaned\n", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    _write_config(config_path)
    data = {
        "groups": {"text": {"files": ["{cleaned_dir}/*.txt"]}},
        "preprocess": {"kind": "cleaner", "config": "config/cleaner.yml"},
        "grouping": {"mode": "auto_single_cleaned"},
    }

    with pytest.raises(ValueError, match="no .txt files"):
        build_run_plan(
            project_root=tmp_path,
            script_dir=None,
            config_path=config_path,
            group_by_file=None,
            auto_single_cleaned=False,
            error_on_empty_group=False,
            load_config_fn=_loader(data),
            preprocess_mode="inspect",
        )

    (cleaned / "a.txt").write_text("a", encoding="utf-8")
    (cleaned / "b.txt").write_text("b", encoding="utf-8")
    with pytest.raises(ValueError, match="expected exactly one"):
        build_run_plan(
            project_root=tmp_path,
            script_dir=None,
            config_path=config_path,
            group_by_file=None,
            auto_single_cleaned=False,
            error_on_empty_group=False,
            load_config_fn=_loader(data),
            preprocess_mode="inspect",
        )


def test_empty_group_policy_and_spec_references(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "whole.txt").write_text("whole", encoding="utf-8")
    (tmp_path / "input" / "part_a.txt").write_text("part_a", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    _write_config(config_path)

    plan = build_run_plan(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=None,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        load_config_fn=_loader({"groups": {"empty": {"files": ["input/no_match*.txt"]}}}),
        preprocess_mode="inspect",
    )
    assert plan.group_files == {"empty": ()}

    with pytest.raises(ValueError, match="No files matched"):
        build_run_plan(
            project_root=tmp_path,
            script_dir=None,
            config_path=config_path,
            group_by_file=None,
            auto_single_cleaned=False,
            error_on_empty_group=True,
            load_config_fn=_loader({"groups": {"empty": {"files": ["input/no_match*.txt"]}}}),
            preprocess_mode="inspect",
        )

    with pytest.raises(ValueError, match="Partition split references empty group: part_b"):
        build_run_plan(
            project_root=tmp_path,
            script_dir=None,
            config_path=config_path,
            group_by_file=None,
            auto_single_cleaned=False,
            error_on_empty_group=False,
            load_config_fn=_loader(
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
        build_run_plan(
            project_root=tmp_path,
            script_dir=None,
            config_path=config_path,
            group_by_file=None,
            auto_single_cleaned=False,
            error_on_empty_group=False,
            load_config_fn=_loader(
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
        load_config=_loader(
            {
                "groups": {
                    "a": {"files": ["input/a.txt"]},
                    "b": {"files": ["input/missing.txt"]},
                },
                "comparisons": [{"name": "ab", "group_a": "a", "group_b": "b"}],
            }
        ),
        cleaner=object(),
        backend_factory=backend_factory,
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
