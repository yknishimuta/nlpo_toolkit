from __future__ import annotations

import ast
from pathlib import Path

from nlpo_toolkit.corpus_analysis.cli import build_parser


def test_deprecated_count_vocabula_package_is_removed() -> None:
    old_package = "count" + "_vocabula"

    assert not Path("nlpo_toolkit", old_package).exists()


def test_no_python_code_imports_removed_count_vocabula_package() -> None:
    forbidden = "nlpo_toolkit." + "count" + "_vocabula"
    offenders: list[Path] = []

    for root in (Path("nlpo_toolkit"), Path("tests")):
        for path in root.rglob("*.py"):
            if path == Path(__file__):
                continue

            text = path.read_text(encoding="utf-8")
            if forbidden in text:
                offenders.append(path)

            tree = ast.parse(text, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module == "nlpo_toolkit":
                        if any(alias.name == "count" + "_vocabula" for alias in node.names):
                            offenders.append(path)
                elif isinstance(node, ast.Import):
                    if any(alias.name == forbidden for alias in node.names):
                        offenders.append(path)

    assert sorted(set(offenders)) == []


def test_count_vocabula_cli_command_remains_registered() -> None:
    parser = build_parser()

    full = parser.parse_args(["count-vocabula", "--project-root", "."])
    short = parser.parse_args(["count", "--project-root", "."])

    assert callable(full.handler)
    assert full.handler is short.handler
