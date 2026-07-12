from __future__ import annotations

import ast
from pathlib import Path


ARCHITECTURE_TEST_ROOT = Path(__file__).resolve().parent


def _effective_body(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.stmt]:
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return body


def _is_noop_statement(node: ast.stmt) -> bool:
    if isinstance(node, ast.Pass):
        return True
    if (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and node.value.value is Ellipsis
    ):
        return True
    return isinstance(node, ast.Return) and node.value is None


def test_architecture_tests_are_not_empty() -> None:
    offenders: list[str] = []
    for path in sorted(ARCHITECTURE_TEST_ROOT.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            body = _effective_body(node)
            if not body or all(_is_noop_statement(item) for item in body):
                offenders.append(f"{path}:{node.lineno}: {node.name}")

    assert offenders == [], "\n".join(offenders)
