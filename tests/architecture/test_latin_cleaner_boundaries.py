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


def test_cli_is_only_an_adapter_and_service_owns_execution() -> None:
    cli = (ROOT / "run_clean_corpus.py").read_text(encoding="utf-8")
    assert not any(name in cli for name in ("clean_document", "load_cleaner_program", "write_ref_events", "append_ref_events"))
    service = (ROOT / "service.py").read_text(encoding="utf-8")
    assert not any(name in service for name in ("import sys", "import argparse", "corpus_analysis", "print("))
    assert not any(name in service for name in ("load_cleaner_config", "inspect_cleaner_config"))


def test_corpus_analysis_uses_the_typed_cleaner_service_contract() -> None:
    contracts = Path("nlpo_toolkit/cleaner_contracts.py").read_text(encoding="utf-8")
    corpus = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("nlpo_toolkit/corpus_analysis").rglob("*.py")
    )
    assert "CleanerRunner" not in contracts
    assert "CleanerLoader" not in contracts
    assert "run_clean_corpus" not in corpus
    assert "main([" not in corpus
    assert not Path("nlpo_toolkit/corpus_analysis/cleaner_runtime.py").exists()
