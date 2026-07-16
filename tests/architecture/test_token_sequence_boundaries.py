from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "nlpo_toolkit/corpus_analysis/token_sequences"
ANALYSIS = ROOT / "nlpo_toolkit/corpus_analysis"


def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }


def definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def string_constants(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_package_exists_and_init_is_only_a_marker():
    assert {"models.py", "grouping.py", "fields.py", "context.py"} <= {
        path.name for path in PACKAGE.iterdir()
    }
    tree = ast.parse((PACKAGE / "__init__.py").read_text(encoding="utf-8"))
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))


def test_consumers_use_the_same_grouping_owner_and_old_helpers_are_absent():
    ngram = ANALYSIS / "ngram.py"
    concordance = ANALYSIS / "concordance.py"
    assert "token_sequences.grouping" in imports(ngram)
    assert "token_sequences.grouping" in imports(concordance)
    removed = {
        "_sequence_key", "_row_from_record", "build_ngrams_from_rows",
        "read_token_artifact_rows", "iter_config_token_rows", "_field_value",
        "_kwic_from_sequence",
    }
    production_definitions = set().union(*(
        definitions(path) for path in ANALYSIS.rglob("*.py")
    ))
    assert removed.isdisjoint(production_definitions)


def test_common_package_has_only_inward_model_dependencies():
    forbidden = ("token_artifact", "ngram", "concordance", "cli", "corpus", "artifact_plan")
    for path in PACKAGE.glob("*.py"):
        module_imports = imports(path)
        assert not any(any(term in module for term in forbidden) for module in module_imports)
    assert imports(PACKAGE / "models.py") == {"__future__", "dataclasses", "types", "typing"}


def test_consumer_ast_has_no_legacy_sequence_mechanics():
    ngram = ANALYSIS / "ngram.py"
    concordance = ANALYSIS / "concordance.py"
    legacy_columns = {"sent_idx", "file", "label", "chunk"}
    assert legacy_columns.isdisjoint(string_constants(ngram))
    tree = ast.parse(concordance.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"index", "split"}
        for node in ast.walk(tree)
    )
    assert "collections" not in imports(concordance)
