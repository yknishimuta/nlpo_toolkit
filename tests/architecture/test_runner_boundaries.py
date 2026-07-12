from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path


RUNNER_MODULES = (
    Path("nlpo_toolkit/corpus_analysis/runner_types.py"),
    Path("nlpo_toolkit/corpus_analysis/runtime.py"),
    Path("nlpo_toolkit/corpus_analysis/analysis_pipeline.py"),
    Path("nlpo_toolkit/corpus_analysis/post_analysis.py"),
    Path("nlpo_toolkit/corpus_analysis/run_reporting.py"),
)


def test_runner_top_level_function_surface_is_only_run() -> None:
    path = Path("nlpo_toolkit/corpus_analysis/runner.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}

    assert functions == {"run"}
    assert classes == set()


def test_runner_responsibilities_live_in_dedicated_modules() -> None:
    from nlpo_toolkit.corpus_analysis.analysis_pipeline import analyze_corpora
    from nlpo_toolkit.corpus_analysis.post_analysis import (
        execute_group_comparisons,
        execute_partition_validations,
    )
    from nlpo_toolkit.corpus_analysis.run_reporting import build_final_run_metadata
    from nlpo_toolkit.corpus_analysis.runtime import prepare_run_context

    assert prepare_run_context.__module__.endswith(".runtime")
    assert analyze_corpora.__module__.endswith(".analysis_pipeline")
    assert execute_partition_validations.__module__.endswith(".post_analysis")
    assert execute_group_comparisons.__module__.endswith(".post_analysis")
    assert build_final_run_metadata.__module__.endswith(".run_reporting")


def test_runner_modules_import_independently() -> None:
    pass


def test_split_modules_do_not_import_runner() -> None:
    violations: list[str] = []
    for path in RUNNER_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "runner" or module.endswith(".runner"):
                    violations.append(f"{path}:{node.lineno}: from {module} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.endswith(".runner"):
                        violations.append(f"{path}:{node.lineno}: import {alias.name}")

    assert violations == []


def test_run_context_contains_plan_and_runtime_state_only() -> None:
    from nlpo_toolkit.corpus_analysis.runner_types import RunContext

    names = {field.name for field in fields(RunContext)}

    assert names == {
        "plan",
        "nlp",
        "backend_info",
        "splitter_nlp",
        "roman_exceptions",
        "extraction_policy",
    }
    assert not names & {
        "project_root",
        "config_path",
        "config",
        "out_dir",
        "grouping_mode",
        "per_file",
        "auto_mode",
        "auto_group_name",
        "work_items",
        "group_files",
        "analysis_unit",
        "csv_header",
    }
