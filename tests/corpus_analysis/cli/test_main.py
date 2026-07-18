from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest
from types import SimpleNamespace

from nlpo_toolkit.corpus_analysis.cli import build_parser, main
from nlpo_toolkit.corpus_analysis.cli import count as count_cli


def test_root_parser_registers_all_commands() -> None:
    parser = build_parser()
    subparsers = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]

    assert len(subparsers) == 1
    assert set(subparsers[0].choices) == {
        "cache",
        "compare",
        "concordance",
        "count",
        "features",
        "ngram",
        "stylometry",
    }


def test_count_uses_count_handler() -> None:
    parser = build_parser()
    args = parser.parse_args(["count", "--project-root", "."])

    assert args.handler is count_cli.execute


def test_count_vocabula_command_is_not_registered() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["count-vocabula", "--project-root", "."])

    assert exc_info.value.code == 2


def test_compare_metric_option_is_not_registered() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            ["compare", "--inputs", "a.csv", "b.csv", "--metric", "log-ratio"]
        )
    assert exc_info.value.code == 2
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert "--metric" not in subparsers.choices["compare"].format_help()


def test_main_dispatches_through_registered_handler(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_execute_count_command(request, *, dependencies):
        calls.append((request, dependencies))
        return SimpleNamespace(
            successful=True,
            archive=None,
            run=SimpleNamespace(partition_mismatches=()),
        )

    monkeypatch.setattr(
        count_cli,
        "execute_count_command",
        fake_execute_count_command,
    )

    rc = main(["count", "--project-root", str(tmp_path)])

    assert rc == 0
    assert calls[0][0].corpus.project_root == tmp_path.resolve()
    assert calls[0][0].command_line == (
        "nlpo",
        "count",
        "--project-root",
        str(tmp_path),
    )


def test_parser_build_does_not_initialize_count_backend(monkeypatch) -> None:
    def fail_command(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("command service must not run while building parser")

    monkeypatch.setattr(count_cli, "execute_count_command", fail_command)

    build_parser()


def test_root_main_has_no_command_name_if_dispatch() -> None:
    source = Path("nlpo_toolkit/corpus_analysis/cli/main.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    main_func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    command_comparisons = [
        node
        for node in ast.walk(main_func)
        if isinstance(node, ast.Compare)
        and any(
            isinstance(left, ast.Attribute) and left.attr == "command"
            for left in [node.left]
        )
    ]

    assert command_comparisons == []
