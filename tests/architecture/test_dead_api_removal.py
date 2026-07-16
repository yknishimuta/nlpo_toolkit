from __future__ import annotations

import ast
import inspect
from dataclasses import fields
from pathlib import Path

from nlpo_toolkit.comparison.cli_service import CompareRequest
from nlpo_toolkit.corpus_analysis.ngram import read_token_artifact_rows


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT / "nlpo_toolkit"


def test_removed_request_and_function_parameters_are_absent() -> None:
    assert "metric" not in {field.name for field in fields(CompareRequest)}
    assert tuple(inspect.signature(read_token_artifact_rows).parameters) == (
        "tokens_path", "field"
    )


def test_removed_symbols_are_not_defined_exported_or_imported() -> None:
    removed = {
        "ComparisonReport", "compare_frequency_tables", "split_frequency_csv",
        "feature_upos_value", "prune_analysis_cache", "PruneReport",
    }
    offenders = []
    for path in PRODUCTION.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in removed:
                offenders.append((path, node.name))
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in removed:
                        offenders.append((path, alias.name))
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            ):
                exported = {
                    item.value for item in ast.walk(node.value)
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                }
                for name in exported & removed:
                    offenders.append((path, name))
    assert offenders == []


def test_compare_cli_has_no_metric_option() -> None:
    tree = ast.parse(
        (PRODUCTION / "corpus_analysis/cli/compare.py").read_text(encoding="utf-8")
    )
    strings = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "--metric" not in strings


def test_removed_maintenance_module_is_absent() -> None:
    assert not (PRODUCTION / "corpus_analysis/analysis_cache/maintenance.py").exists()
