from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "nlpo_toolkit/corpus_analysis/analysis_cache"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }


def test_old_module_is_replaced_by_non_facade_package() -> None:
    assert not (PACKAGE.parent / "analysis_cache.py").exists()
    assert PACKAGE.is_dir()
    init_tree = ast.parse((PACKAGE / "__init__.py").read_text(encoding="utf-8"))
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in init_tree.body)


def test_cache_package_dependency_direction() -> None:
    forbidden = {
        "models.py": {"repository", "service", "writer", "maintenance"},
        "keys.py": {"repository", "service", "writer"},
        "codec.py": {"repository", "service"},
        "writer.py": {"repository", "service"},
        "repository.py": {"service", "locking"},
        "maintenance.py": {"service", "repository", "writer"},
        "stats.py": {"repository", "service", "writer"},
    }
    for filename, names in forbidden.items():
        imports = _imports(PACKAGE / filename)
        assert not any(module.rsplit(".", 1)[-1] in names for module in imports), filename


def test_raw_lock_primitives_are_owned_only_by_locking() -> None:
    for path in PACKAGE.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if path.name == "locking.py":
            assert "acquire_cache_lock" in source and "release_cache_lock" in source
        else:
            assert "acquire_cache_lock" not in source
            assert "release_cache_lock" not in source


def test_cache_symbols_have_single_owners() -> None:
    owners = {
        "open_or_compute_analysis_records": "service.py",
        "prune_analysis_cache": "maintenance.py",
        "read_cache_metadata": "codec.py",
        "read_analysis_records": "codec.py",
        "validate_cache_object": "codec.py",
        "AnalysisCacheWriter": "writer.py",
    }
    for symbol, owner in owners.items():
        definitions = []
        for path in PACKAGE.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if any(
                isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == symbol
                for node in tree.body
            ):
                definitions.append(path.name)
        assert definitions == [owner]


def test_analysis_execution_has_no_manual_cache_iterator_close() -> None:
    source = (PACKAGE.parent / "analysis_execution.py").read_text(encoding="utf-8")
    assert "getattr(source.records" not in source
    assert "stack.callback(close)" not in source
