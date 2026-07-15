from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

from nlpo_toolkit.corpus_analysis.analysis_results import (
    AnalysisResults,
    GroupAnalysisResult,
)


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT / "nlpo_toolkit/corpus_analysis"


def test_analysis_result_fields_have_one_canonical_group_store() -> None:
    assert {field.name for field in fields(AnalysisResults)} == {"groups", "cache_stats"}
    group_fields = {field.name for field in fields(GroupAnalysisResult)}
    assert "label" not in group_fields
    assert group_fields == {
        "files", "counter", "ref_tag_counts", "output_files",
        "trace_path", "token_artifact",
    }


def test_production_has_no_legacy_analysis_result_access() -> None:
    forbidden_attributes = {
        "counters_by_group", "files_by_group", "ref_tags_by_group",
        "trace_paths", "token_artifacts", "analysis_cache",
    }
    offenders: list[str] = []
    for path in PRODUCTION.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "analysis"
                and node.attr in forbidden_attributes
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert offenders == []


def test_old_duplicate_builder_is_removed() -> None:
    tree = ast.parse(
        (PRODUCTION / "analysis_orchestration.py").read_text(encoding="utf-8")
    )
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_build_analysis_results"
        for node in tree.body
    )
