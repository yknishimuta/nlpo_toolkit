from __future__ import annotations

import ast
from pathlib import Path


CONFIG_REFERENCES = Path("nlpo_toolkit/corpus_analysis/config_references.py")


def test_config_references_does_not_import_yaml_or_read_files() -> None:
    tree = ast.parse(
        CONFIG_REFERENCES.read_text(encoding="utf-8"),
        filename=str(CONFIG_REFERENCES),
    )
    forbidden_calls: list[tuple[int, str]] = []
    yaml_imports: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "yaml" for alias in node.names):
                yaml_imports.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module == "yaml":
            yaml_imports.append(node.lineno)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"read_text", "read_bytes", "open", "safe_load", "load"}:
                forbidden_calls.append((node.lineno, node.func.attr))
    assert yaml_imports == []
    assert forbidden_calls == []


def test_config_references_does_not_interpret_cleaner_keys() -> None:
    tree = ast.parse(CONFIG_REFERENCES.read_text(encoding="utf-8"))
    forbidden = {"input", "output", "rules_path", "lexicon_map_path", "ref_tsv"}
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert literals.isdisjoint(forbidden)


def test_archive_and_preprocess_do_not_parse_cleaner_yaml() -> None:
    for path in (
        Path("nlpo_toolkit/corpus_analysis/archive.py"),
        Path("nlpo_toolkit/corpus_analysis/preprocess.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert "yaml" not in source
        assert "resolve_cleaner_output_dir" not in source
