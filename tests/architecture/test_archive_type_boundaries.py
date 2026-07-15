from __future__ import annotations

import ast
import inspect
from dataclasses import fields
from pathlib import Path

from nlpo_toolkit.corpus_analysis import archive_types, config_references
import nlpo_toolkit.corpus_analysis.archive.models as models
from nlpo_toolkit.corpus_analysis.archive.service import create_run_archive
from nlpo_toolkit.corpus_analysis.ports import ArchiveCreator
from nlpo_toolkit.corpus_analysis.runner_types import RunResult


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT / "nlpo_toolkit"


def test_removed_archive_types_and_fields_are_absent() -> None:
    for module, names in (
        (archive_types, {"ArchiveOptions"}),
        (models, {"ArchiveFile"}),
        (config_references, {"ReferencedConfigFile"}),
    ):
        assert not any(hasattr(module, name) for name in names)
    assert {field.name for field in fields(config_references.ConfigFileReference)} == {
        "kind",
        "source_path",
        "archive_policy",
        "snapshot_relative_path",
    }
    assert {field.name for field in fields(RunResult)} >= {"config_references"}
    assert "config_files" not in {field.name for field in fields(RunResult)}


def test_archive_contract_names_are_explicit() -> None:
    assert set(config_references.ConfigArchivePolicy) == {
        config_references.ConfigArchivePolicy.SNAPSHOT,
        config_references.ConfigArchivePolicy.METADATA_ONLY,
    }
    assert {field.name for field in fields(models.ArchivedFile)} == {
        "source_path",
        "archive_relative_path",
        "sha256",
        "size_bytes",
    }
    assert {field.name for field in fields(archive_types.RunArchiveResult)} == {
        "archive_directory",
        "copied_files",
    }
    assert tuple(inspect.signature(create_run_archive).parameters) == (
        "run_result",
        "request",
    )
    assert tuple(inspect.signature(ArchiveCreator.__call__).parameters)[1:] == (
        "run_result",
        "request",
    )


def test_production_has_no_compatibility_aliases_or_boolean_archive_policy() -> None:
    forbidden_names = {
        "ReferencedConfigFile",
        "ArchiveOptions",
        "ArchiveFile",
        "copy_to_snapshot",
    }
    offenders: list[str] = []
    for path in PRODUCTION.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Name, ast.Attribute, ast.arg)):
                name = (
                    node.id
                    if isinstance(node, ast.Name)
                    else node.attr
                    if isinstance(node, ast.Attribute)
                    else node.arg
                )
                if name in forbidden_names:
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}:{name}")
    assert offenders == []


def test_archive_does_not_reinterpret_config_or_resolve_project_paths() -> None:
    for path in (PRODUCTION / "corpus_analysis/archive").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert "enabled" not in attributes
        assert "config" not in attributes
        assert not any(isinstance(node, ast.FunctionDef) and node.name == "__getattr__" for node in tree.body)
