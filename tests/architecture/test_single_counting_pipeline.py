from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import nlpo_toolkit.nlp as nlp
from nlpo_toolkit.corpus_analysis.ports import RunnerDependencies


def test_analysis_execution_does_not_call_legacy_counter() -> None:
    path = Path("nlpo_toolkit/corpus_analysis/runner.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    old_attr = "count" + "_group"
    old_arg = "count" + "_group_fn"

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == old_attr:
            violations.append(f"{path}:{node.lineno}: legacy counter attribute access")
        if isinstance(node, ast.Name) and node.id == old_arg:
            violations.append(f"{path}:{node.lineno}: legacy counter argument reference")

    assert violations == []


def test_production_code_does_not_import_removed_nlp_hook_module() -> None:
    offenders: list[Path] = []
    forbidden = "nlp" + "_hooks"

    for path in Path("nlpo_toolkit").rglob("*.py"):
        if forbidden in path.read_text(encoding="utf-8"):
            offenders.append(path)

    assert offenders == []


def test_runner_dependencies_has_no_counter_injection() -> None:
    names = {field.name for field in fields(RunnerDependencies)}

    assert ("count" + "_group") not in names


def test_legacy_streaming_counters_are_removed() -> None:
    assert not hasattr(nlp, "count" + "_nouns_streaming")
