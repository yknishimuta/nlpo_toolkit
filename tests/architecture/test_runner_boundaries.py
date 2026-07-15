from __future__ import annotations

import ast
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUNNER_SPLIT_MODULE_NAMES = (
    "nlpo_toolkit.corpus_analysis.runner_types",
    "nlpo_toolkit.corpus_analysis.analysis_policy",
    "nlpo_toolkit.corpus_analysis.analysis_records",
    "nlpo_toolkit.corpus_analysis.runtime",
    "nlpo_toolkit.corpus_analysis.analysis_execution",
    "nlpo_toolkit.corpus_analysis.analysis_outputs",
    "nlpo_toolkit.corpus_analysis.analysis_orchestration",
    "nlpo_toolkit.corpus_analysis.post_analysis",
    "nlpo_toolkit.corpus_analysis.run_reporting",
)
RUNNER_RELATED_MODULE_NAMES = (
    "nlpo_toolkit.corpus_analysis.ports",
    "nlpo_toolkit.corpus_analysis.composition",
    *RUNNER_SPLIT_MODULE_NAMES,
)
RUNNER_IMPORTABLE_MODULE_NAMES = (
    *RUNNER_RELATED_MODULE_NAMES,
    "nlpo_toolkit.corpus_analysis.runner",
)


def _source_path(module_name: str) -> Path:
    return PROJECT_ROOT / Path(*module_name.split(".")).with_suffix(".py")


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
    from nlpo_toolkit.corpus_analysis.analysis_orchestration import analyze_corpora
    from nlpo_toolkit.corpus_analysis.post_analysis import (
        execute_group_comparisons,
        execute_partition_validations,
    )
    from nlpo_toolkit.corpus_analysis.run_reporting import build_final_run_metadata
    from nlpo_toolkit.corpus_analysis.runtime import prepare_run_context

    assert prepare_run_context.__module__.endswith(".runtime")
    assert analyze_corpora.__module__.endswith(".analysis_orchestration")
    assert execute_partition_validations.__module__.endswith(".post_analysis")
    assert execute_group_comparisons.__module__.endswith(".post_analysis")
    assert build_final_run_metadata.__module__.endswith(".run_reporting")


@pytest.mark.parametrize(
    "module_name",
    RUNNER_IMPORTABLE_MODULE_NAMES,
    ids=lambda name: name.rsplit(".", 1)[-1],
)
def test_runner_modules_import_independently(module_name: str) -> None:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    pythonpath = [str(PROJECT_ROOT)]
    if existing_pythonpath:
        pythonpath.append(existing_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib; importlib.import_module({module_name!r})",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )

    assert completed.returncode == 0, (
        f"Command: {completed.args!r}\n"
        f"Module: {module_name}\n"
        f"Return code: {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def test_split_modules_do_not_import_runner() -> None:
    violations: list[str] = []
    for module_name in RUNNER_SPLIT_MODULE_NAMES:
        path = _source_path(module_name)
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

    assert names == {"session", "sentence_splitter"}
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
