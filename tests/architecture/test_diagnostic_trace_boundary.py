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
    for name in ("token_artifact.py", "concordance.py", "ngram.py"):
        path = Path("nlpo_toolkit/corpus_analysis") / name
        assert "diagnostic_trace" not in _imports(path), path
        assert "read_legacy_trace_records" not in path.read_text(encoding="utf-8")


def test_diagnostic_trace_is_writer_only() -> None:
    import nlpo_toolkit.corpus_analysis.diagnostic_trace as trace

    assert hasattr(trace, "DiagnosticTraceWriter")
    assert not hasattr(trace, "read_legacy_trace_records")


def test_token_artifact_has_no_trace_fallback() -> None:
    import nlpo_toolkit.corpus_analysis.token_artifact as artifact

    assert not hasattr(artifact, "read_token_rows")


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
