from __future__ import annotations

import ast
import inspect
from pathlib import Path

from nlpo_toolkit.corpus_analysis.io_utils import read_concat


def test_read_concat_has_no_broad_exception_handler_or_continue() -> None:
    tree = ast.parse(inspect.getsource(read_concat))
    broad_handlers = []
    continues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Continue):
            continues.append(node.lineno)
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            broad_handlers.append(node.lineno)
        elif isinstance(node.type, ast.Name) and node.type.id in {
            "Exception",
            "BaseException",
        }:
            broad_handlers.append(node.lineno)

    assert broad_handlers == []
    assert continues == []


def test_old_read_warning_was_removed() -> None:
    source = Path("nlpo_toolkit/corpus_analysis/io_utils.py").read_text(
        encoding="utf-8"
    )

    assert "[WARN] failed to read" not in source
    assert "sys.stderr" not in source
    assert "print(" not in source
