import ast
from pathlib import Path

import pytest


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }


def test_downstream_modules_do_not_import_diagnostic_trace() -> None:
    paths = [
        *Path("nlpo_toolkit/corpus_analysis/token_artifact").glob("*.py"),
        Path("nlpo_toolkit/corpus_analysis/concordance.py"),
        Path("nlpo_toolkit/corpus_analysis/ngram.py"),
    ]
    for path in paths:
        assert "diagnostic_trace" not in _imports(path), path
        assert "read_legacy_trace_records" not in path.read_text(encoding="utf-8")


def test_diagnostic_trace_is_writer_only() -> None:
    import nlpo_toolkit.corpus_analysis.diagnostic_trace as trace

    assert hasattr(trace, "DiagnosticTraceWriter")
    assert not hasattr(trace, "read_legacy_trace_records")


def test_token_artifact_has_no_trace_fallback() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("nlpo_toolkit/corpus_analysis/token_artifact").glob("*.py")
    )
    assert "read_token_rows" not in source


def test_downstream_cli_uses_tokens_and_rejects_trace() -> None:
    from nlpo_toolkit.corpus_analysis.cli import build_parser

    parser = build_parser()
    concordance = parser.parse_args(
        ["concordance", "--tokens", "output/tokens.tsv", "--keys", "arma"]
    )
    ngram = parser.parse_args(["ngram", "--tokens", "output/tokens.tsv"])

    assert concordance.tokens == Path("output/tokens.tsv")
    assert not hasattr(concordance, "trace")
    assert ngram.tokens == Path("output/tokens.tsv")
    assert not hasattr(ngram, "trace")
    with pytest.raises(SystemExit):
        parser.parse_args(["concordance", "--trace", "output/trace.tsv", "--keys", "arma"])
    with pytest.raises(SystemExit):
        parser.parse_args(["ngram", "--trace", "output/trace.tsv"])
    with pytest.raises(SystemExit):
        parser.parse_args(["ngram"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "ngram",
                "--tokens",
                "output/tokens.tsv",
                "--config",
                "config/groups.config.yml",
            ]
        )
