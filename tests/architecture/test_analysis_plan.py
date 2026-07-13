from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import get_type_hints

from nlpo_toolkit.corpus_analysis.run_plan import AnalysisPlan
from nlpo_toolkit.corpus_analysis.runner_types import RunContext, RunResult


RUN_PLAN_PATH = Path("nlpo_toolkit/corpus_analysis/run_plan.py")


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }


def test_only_one_plan_dataclass_exists() -> None:
    tree = ast.parse(
        RUN_PLAN_PATH.read_text(encoding="utf-8"),
        filename=str(RUN_PLAN_PATH),
    )
    class_names = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }
    dataclass_names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            (isinstance(decorator, ast.Name) and decorator.id == "dataclass")
            or (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "dataclass"
            )
            for decorator in node.decorator_list
        )
    }

    assert "CorpusPlan" not in class_names
    assert "RunPlan" not in class_names
    assert dataclass_names == {"AnalysisPlan"}
    assert is_dataclass(AnalysisPlan)
    assert AnalysisPlan.__dataclass_params__.frozen is True


def test_analysis_plan_does_not_store_derived_values() -> None:
    field_names = {field.name for field in fields(AnalysisPlan)}
    assert field_names == {
        "project_root",
        "config_path",
        "config",
        "cleaned_dir",
        "grouping_mode",
        "work_items",
        "group_files",
        "cleaner_inspection",
        "config_files",
    }
    assert {
        "per_file",
        "auto_mode",
        "out_dir",
        "auto_group_name",
        "partition_specs",
        "comparison_specs",
        "analysis_unit",
        "use_lemma",
        "csv_header",
    }.isdisjoint(field_names)


def test_consumers_use_the_correct_planning_stage() -> None:
    runtime_imports = _imported_names(
        Path("nlpo_toolkit/corpus_analysis/runtime.py")
    )
    dry_run_imports = _imported_names(
        Path("nlpo_toolkit/corpus_analysis/dry_run.py")
    )
    features_imports = _imported_names(
        Path("nlpo_toolkit/corpus_analysis/features.py")
    )
    ngram_imports = _imported_names(
        Path("nlpo_toolkit/corpus_analysis/ngram.py")
    )

    assert "build_count_plan" in runtime_imports
    assert "build_count_plan" in dry_run_imports
    assert "build_analysis_plan" in features_imports
    assert "build_analysis_plan" in ngram_imports
    assert "build_count_plan" not in features_imports
    assert "build_count_plan" not in ngram_imports


def test_build_count_plan_validates_without_reconstructing_plan() -> None:
    tree = ast.parse(
        RUN_PLAN_PATH.read_text(encoding="utf-8"),
        filename=str(RUN_PLAN_PATH),
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_count_plan"
    )
    called_names = {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]

    assert "build_analysis_plan" in called_names
    assert "validate_count_plan" in called_names
    assert "AnalysisPlan" not in called_names
    assert "replace" not in called_names
    assert len(returns) == 1
    assert isinstance(returns[0].value, ast.Name)
    assert returns[0].value.id == "plan"


def test_runtime_result_types_use_analysis_plan() -> None:
    assert get_type_hints(RunContext)["plan"] is AnalysisPlan
    assert get_type_hints(RunResult)["plan"] is AnalysisPlan
