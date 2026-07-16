from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT / "nlpo_toolkit"
COMMON = PRODUCTION / "configuration/yaml_loader.py"
DOMAIN_LOADERS = (
    PRODUCTION / "corpus_analysis/config/parser.py",
    PRODUCTION / "corpus_analysis/dry_run.py",
    PRODUCTION / "corpus_analysis/cache.py",
    PRODUCTION / "latin/cleaners/config_loader.py",
    PRODUCTION / "latin/cleaners/rule_loader.py",
    PRODUCTION / "latin/latin_wordlist/build_latin_wordlist.py",
)


def _yaml_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "yaml"
        and node.func.attr in {"load", "safe_load"}
    ]


def test_only_common_loader_uses_pyyaml_loading_api() -> None:
    offenders = []
    safe_loader_subclasses = []
    for path in PRODUCTION.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if _yaml_calls(tree) and path != COMMON:
            offenders.append(path.relative_to(ROOT))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(
                isinstance(base, ast.Attribute)
                and isinstance(base.value, ast.Name)
                and base.value.id == "yaml"
                and base.attr == "SafeLoader"
                for base in node.bases
            ):
                safe_loader_subclasses.append(path)
    assert offenders == []
    assert safe_loader_subclasses == [COMMON]


def test_removed_duplicate_and_cache_loaders_are_absent() -> None:
    forbidden = {
        "DuplicateKeyLoader", "DuplicateKeyConfig", "DryRunConfigError",
        "_load_yaml_with_duplicate_keys", "_load_yaml_mapping",
    }
    names = set()
    for path in PRODUCTION.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names.update(
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        )
    assert names.isdisjoint(forbidden)


def test_domain_yaml_loaders_do_not_read_files_directly() -> None:
    for path in DOMAIN_LOADERS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_text"
        ]
        if path.name == "build_latin_wordlist.py":
            # Its remaining reads consume corpus text and wordlists, not YAML.
            assert all(node.lineno > 150 for node in calls)
        else:
            assert calls == []


def test_common_loader_has_no_domain_dependencies() -> None:
    tree = ast.parse(COMMON.read_text(encoding="utf-8"))
    imports = [
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]
    assert not any(
        module.startswith(("nlpo_toolkit.corpus_analysis", "nlpo_toolkit.latin"))
        for module in imports
    )
