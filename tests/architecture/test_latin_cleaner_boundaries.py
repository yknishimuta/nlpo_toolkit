from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path("nlpo_toolkit/latin/cleaners")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}


def test_old_cleaner_module_is_removed_and_init_is_not_facade() -> None:
    assert not (ROOT / "cleaners.py").exists()
    assert not any(isinstance(node, ast.ImportFrom) for node in ast.walk(ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))))


def test_pure_modules_have_no_io_or_corpus_specific_rules() -> None:
    for name in ("pipeline.py", "rule_engine.py"):
        source = (ROOT / name).read_text(encoding="utf-8")
        assert not any(token in source for token in ("read_text", "write_text", ".open(", "mkdir", "yaml", "csv", "print("))
        assert not any(token in source for token in ("HEADER_HASH_RE", "corpus_corporum.yml", "scholastic_text.yml", "#####"))


def test_cleaner_package_has_no_legacy_typing_or_api_definitions() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in ROOT.rglob("*.py"))
    assert not any(token in source for token in ("typing import Any", "\bDict\b", "\bList\b", "\bOptional\b"))
    forbidden = {"clean_text", "clean_corpus_corporum_text", "clean_scholastic_text", "load_patterns_from_yaml", "load_lexicon_map_tsv", "flatten_ref", "_write_ref_events_tsv"}
    definitions = {node.name for path in ROOT.rglob("*.py") for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert definitions.isdisjoint(forbidden)


def test_runner_loads_program_only_outside_single_file_function() -> None:
    tree = ast.parse((ROOT / "run_clean_corpus.py").read_text(encoding="utf-8"))
    single = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_clean_single_file")
    assert "load_cleaner_program" not in {node.func.id for node in ast.walk(single) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
