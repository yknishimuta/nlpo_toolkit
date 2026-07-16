from __future__ import annotations

import ast
from pathlib import Path


def test_summary_renderer_has_no_filesystem_write_calls() -> None:
    path = Path("nlpo_toolkit/corpus_analysis/reporting/summary.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called.isdisjoint({"open", "write_text", "mkdir", "replace"})


def test_removed_reporting_modules_are_absent() -> None:
    root = Path("nlpo_toolkit/corpus_analysis")
    assert not (root / "analysis_outputs.py").exists()
    assert not (root / "outputs.py").exists()
    assert not (root / "run_reporting.py").exists()
