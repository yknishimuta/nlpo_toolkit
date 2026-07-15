from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "nlpo_toolkit/corpus_analysis/archive"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_old_archive_module_is_replaced_by_non_facade_package() -> None:
    assert not (PACKAGE.parent / "archive.py").exists()
    assert PACKAGE.is_dir()
    tree = ast.parse(_source(PACKAGE / "__init__.py"))
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body)


def test_archive_symbols_have_single_owners() -> None:
    owners = {
        "collect_archive_inventory": "inventory.py",
        "copy_archive_inventory": "copying.py",
        "build_archive_manifest": "manifest.py",
        "create_run_archive": "service.py",
        "ArchivedFile": "models.py",
        "RunArchiveError": "errors.py",
    }
    for symbol, owner in owners.items():
        definitions = []
        for path in PACKAGE.glob("*.py"):
            tree = ast.parse(_source(path))
            if any(
                isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == symbol
                for node in tree.body
            ):
                definitions.append(path.name)
        assert definitions == [owner]


def test_archive_package_does_not_discover_or_reinterpret_inputs() -> None:
    forbidden_calls = {"glob", "rglob", "walk"}
    forbidden_text = {
        "expand_globs", "expand_cleaned_dir_placeholders",
        "resolve_cleaner_output_dir", "safe_load",
    }
    for path in PACKAGE.glob("*.py"):
        source = _source(path)
        assert not any(fragment in source for fragment in forbidden_text), path.name
        tree = ast.parse(source)
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in forbidden_calls
            for node in ast.walk(tree)
        ), path.name


def test_manifest_is_filesystem_and_process_independent() -> None:
    source = _source(PACKAGE / "manifest.py")
    forbidden = {"subprocess", "copy2", ".stat(", ".open(", ".glob(", ".rglob("}
    assert not any(fragment in source for fragment in forbidden)
    assert "RunResult" not in source


def test_service_contains_only_orchestration_and_rollback_details() -> None:
    source = _source(PACKAGE / "service.py")
    assert "copy2" not in source
    assert "sha256" not in source
    assert "subprocess.run" not in source
    assert '"run_name":' not in source
