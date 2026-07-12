from __future__ import annotations

import ast
from pathlib import Path


DRY_RUN_PATH = Path("nlpo_toolkit/corpus_analysis/dry_run.py")


def test_dry_run_has_no_broad_exception_handlers() -> None:
    tree = ast.parse(
        DRY_RUN_PATH.read_text(encoding="utf-8"),
        filename=str(DRY_RUN_PATH),
    )
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            offenders.append(f"{DRY_RUN_PATH}:{node.lineno}: bare except")
        elif isinstance(node.type, ast.Name) and node.type.id in {
            "Exception",
            "BaseException",
        }:
            offenders.append(
                f"{DRY_RUN_PATH}:{node.lineno}: except {node.type.id}"
            )
    assert offenders == []


def test_dry_run_orchestration_does_not_catch_low_level_errors() -> None:
    tree = ast.parse(
        DRY_RUN_PATH.read_text(encoding="utf-8"),
        filename=str(DRY_RUN_PATH),
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute_dry_run"
    )
    forbidden = {
        "ValueError",
        "OSError",
        "UnicodeError",
        "YAMLError",
        "Exception",
        "BaseException",
    }
    caught: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.ExceptHandler) or node.type is None:
            continue
        if isinstance(node.type, ast.Name):
            caught.add(node.type.id)
        elif isinstance(node.type, ast.Tuple):
            caught.update(
                item.id for item in node.type.elts if isinstance(item, ast.Name)
            )
    assert caught.isdisjoint(forbidden)
