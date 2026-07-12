from __future__ import annotations

import argparse
from pathlib import Path

from nlpo_toolkit.corpus_analysis.cli import build_parser


def test_only_count_command_is_registered() -> None:
    parser = build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    assert "count" in subparsers.choices
    assert "count-vocabula" not in subparsers.choices


def test_production_has_no_count_vocabula_name() -> None:
    fragments = (
        "count-vocabula",
        "count_vocabula",
        "run_count_vocabula",
        "dry_run_count_vocabula",
    )
    offenders = [
        (str(path), fragment)
        for path in Path("nlpo_toolkit").rglob("*.py")
        for fragment in fragments
        if fragment in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_old_count_vocabula_package_is_absent() -> None:
    assert not Path("nlpo_toolkit/count_vocabula").exists()


def test_readme_uses_canonical_count_command() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "nlpo count-vocabula" not in readme
    assert "## Count Vocabula CLI" not in readme
    assert "nlpo count" in readme
    assert "## Count CLI" in readme
