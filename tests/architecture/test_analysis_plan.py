from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import get_type_hints

from nlpo_toolkit.corpus_analysis.planning.models import (
    AnalysisMode,
    AnalysisPlan,
    ResolvedAnalysisPlan,
)
from nlpo_toolkit.corpus_analysis.runner_types import RunResult


ROOT = Path("nlpo_toolkit/corpus_analysis")
PLANNING = ROOT / "planning"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def _calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_planning_package_replaces_run_plan_module() -> None:
    assert not (ROOT / "run_plan.py").exists()
    assert {path.name for path in PLANNING.glob("*.py")} >= {
        "__init__.py", "models.py", "build.py", "resolve.py", "validate.py"
    }


def test_plan_models_are_frozen_and_do_not_duplicate_static_state() -> None:
    for model in (AnalysisMode, AnalysisPlan, ResolvedAnalysisPlan):
        assert is_dataclass(model)
        assert model.__dataclass_params__.frozen is True
    assert {field.name for field in fields(AnalysisPlan)} == {
        "project_root", "config_path", "config", "out_dir", "grouping_mode",
        "error_on_empty_group", "analysis_mode", "cleaner_plan", "config_files",
    }
    assert {field.name for field in fields(ResolvedAnalysisPlan)} == {
        "definition", "cleaned_dir", "work_items", "group_files"
    }
    assert "cleaner_inspection" not in {field.name for field in fields(AnalysisPlan)}
    for proxy in (
        "project_root", "config_path", "config", "config_files", "grouping_mode",
        "per_file", "auto_mode", "auto_group_name", "out_dir", "partition_specs",
        "comparison_specs", "analysis_unit", "use_lemma", "csv_header",
    ):
        assert not hasattr(ResolvedAnalysisPlan, proxy)
    assert get_type_hints(RunResult)["plan"] is ResolvedAnalysisPlan


def test_planning_import_direction_is_acyclic() -> None:
    models = _imports(PLANNING / "models.py")
    validate = _imports(PLANNING / "validate.py")
    build = _imports(PLANNING / "build.py")
    resolve = _imports(PLANNING / "resolve.py")
    assert not any(name.endswith((".build", ".resolve", ".validate", ".preprocessing")) for name in models)
    assert not any(name.endswith((".build", ".resolve", ".preprocessing")) for name in validate)
    assert not any(name.endswith((".resolve", ".preprocessing")) for name in build)
    assert not any(name.endswith(".build") for name in resolve)
    preprocessing = _imports(ROOT / "preprocessing.py")
    assert not any(name.endswith((".build", ".resolve", ".validate")) for name in preprocessing)


def test_stage_modules_do_not_cross_responsibility_boundaries() -> None:
    assert _calls(PLANNING / "build.py").isdisjoint(
        {"execute_preprocess", "execute_cleaner", "resolve_corpus_work_items", "prepare_corpora"}
    )
    assert _calls(PLANNING / "validate.py").isdisjoint(
        {"open", "read_text", "write_text", "exists", "glob", "execute_cleaner"}
    )
    assert "load_config" not in _calls(PLANNING / "resolve.py")


def test_no_production_imports_removed_run_plan() -> None:
    offenders = []
    for path in Path("nlpo_toolkit").rglob("*.py"):
        if any("run_plan" in module for module in _imports(path)):
            offenders.append(str(path))
    assert offenders == []


def test_dry_run_uses_build_and_inspection_without_preparation_dependencies() -> None:
    source = (ROOT / "dry_run.py").read_text(encoding="utf-8")
    assert "build_count_plan" in source
    assert "inspect_analysis_plan" in source
    assert "CorpusPreparationDependencies" not in source
    assert "execute_preprocess" not in source
