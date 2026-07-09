from __future__ import annotations


def test_old_cli_import_aliases_canonical_cli() -> None:
    from nlpo_toolkit.count_vocabula import cli as old_cli
    from nlpo_toolkit.corpus_analysis import cli as new_cli

    assert old_cli.main is new_cli.main
    assert old_cli.build_parser is new_cli.build_parser
