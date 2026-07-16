from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path("nlpo_toolkit/comparison")


def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.module or "" for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }


def definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef))}


def test_old_modules_are_deleted_and_new_owners_exist():
    for name in ("configured.py", "cli_service.py", "models.py", "writers.py"):
        assert not (ROOT / name).exists()
    for name in ("config.py", "errors.py", "engine.py", "results.py", "frequency_io.py"):
        assert (ROOT / name).is_file()
    assert (ROOT / "services/configured.py").is_file()
    assert (ROOT / "services/csv.py").is_file()


def test_package_initializers_are_markers():
    for path in (ROOT / "__init__.py", ROOT / "services/__init__.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))


def test_layer_import_directions():
    assert not any(term in module for module in imports(ROOT / "config.py") for term in ("engine", "results", "services", "pathlib"))
    assert not any(term in module for module in imports(ROOT / "results.py") for term in ("config", "services", "cli", "pathlib"))
    assert not any(term in module for module in imports(ROOT / "engine.py") for term in ("config", "services", "frequency_io", "cli", "pathlib"))
    assert not any(term in module for module in imports(ROOT / "services/configured.py") for term in ("frequency_io", "artifact", "corpus_analysis"))
    assert not any(term in module for module in imports(ROOT / "services/csv.py") for term in ("comparison.config", "corpus_analysis", "artifact"))


def test_removed_types_and_helpers_are_not_defined():
    removed = {"ComparisonRow", "ComparisonResult", "CompareRequest", "CompareCommandResult", "comparison_result_summary", "comparison_result_meta", "comparison_csv_name", "sanitize_comparison_name", "_render_pair_rows", "_render_many_rows"}
    found = set().union(*(definitions(path) for path in ROOT.rglob("*.py")))
    assert removed.isdisjoint(found)
