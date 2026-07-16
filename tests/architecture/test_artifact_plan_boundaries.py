from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

from nlpo_toolkit.corpus_analysis.artifacts.models import ArtifactPlan
from nlpo_toolkit.corpus_analysis.runner_types import RunResult


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "nlpo_toolkit/corpus_analysis"


def test_removed_output_inventories_do_not_return() -> None:
    production = "\n".join(
        path.read_text(encoding="utf-8") for path in PACKAGE.rglob("*.py")
    )
    for removed in ("AnalysisOutputPlan", "GroupOutputPaths", "merge_generated_outputs"):
        assert removed not in production


def test_only_artifact_plan_stores_generated_artifact_tuple() -> None:
    assert {field.name for field in fields(ArtifactPlan)} == {"artifacts"}
    run_fields = {field.name for field in fields(RunResult)}
    assert "artifact_plan" in run_fields
    assert not {"output_files", "trace_files", "summary_path", "metadata_path"} & run_fields


def test_writers_and_reporting_do_not_construct_count_output_paths() -> None:
    forbidden = {
        "post_analysis.py": ("partition_validation_", "group_comparison_", "group_comparisons.json"),
        "reporting/service.py": ("summary.txt", "run_meta.json"),
    }
    for filename, fragments in forbidden.items():
        source = (PACKAGE / filename).read_text(encoding="utf-8")
        string_literals = tuple(
            node.value for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
        for fragment in fragments:
            assert not any(fragment in literal for literal in string_literals)


def test_result_dataclasses_have_no_generated_outputs_field() -> None:
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name != "ArtifactPlan":
                annotated = {
                    item.target.id for item in node.body
                    if isinstance(item, ast.AnnAssign)
                    and isinstance(item.target, ast.Name)
                }
                assert "generated_outputs" not in annotated
