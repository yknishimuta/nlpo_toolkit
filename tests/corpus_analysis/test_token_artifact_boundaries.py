from __future__ import annotations

import ast
import csv
import subprocess
import sys
from pathlib import Path

from nlpo_toolkit.corpus_analysis.analysis_records import TokenRecord
from nlpo_toolkit.corpus_analysis.token_artifact import (
    TOKEN_ARTIFACT_COLUMNS,
    TokenArtifactMetadata,
    TokenArtifactWriter,
)


def test_token_artifact_compatibility_reexports_are_identical() -> None:
    from nlpo_toolkit.corpus_analysis import analysis_records
    from nlpo_toolkit.corpus_analysis import diagnostic_trace
    from nlpo_toolkit.corpus_analysis import token_artifact

    assert token_artifact.NLPAnalysisRecord is analysis_records.NLPAnalysisRecord
    assert token_artifact.TokenRecord is analysis_records.TokenRecord
    assert token_artifact.AnalysisOptions is analysis_records.AnalysisOptions
    assert (
        token_artifact.evaluate_analysis_record
        is analysis_records.evaluate_analysis_record
    )
    assert token_artifact.DiagnosticTraceWriter is diagnostic_trace.DiagnosticTraceWriter
    assert token_artifact.LEGACY_TRACE_COLUMNS is diagnostic_trace.LEGACY_TRACE_COLUMNS


def test_analysis_cache_imports_analysis_record_from_canonical_module() -> None:
    from nlpo_toolkit.corpus_analysis import analysis_cache
    from nlpo_toolkit.corpus_analysis import analysis_records

    assert analysis_cache.NLPAnalysisRecord is analysis_records.NLPAnalysisRecord


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            prefix = "." * node.level
            modules.add(f"{prefix}{node.module}")
    return modules


def test_analysis_cache_does_not_import_token_artifact_or_trace() -> None:
    modules = _imported_modules(Path("nlpo_toolkit/corpus_analysis/analysis_cache.py"))

    assert ".token_artifact" not in modules
    assert "nlpo_toolkit.corpus_analysis.token_artifact" not in modules
    assert ".diagnostic_trace" not in modules
    assert "nlpo_toolkit.corpus_analysis.diagnostic_trace" not in modules


def test_analysis_records_has_no_artifact_cache_or_pipeline_dependencies() -> None:
    modules = _imported_modules(Path("nlpo_toolkit/corpus_analysis/analysis_records.py"))

    forbidden = {
        ".token_artifact",
        ".diagnostic_trace",
        ".analysis_cache",
        ".runner",
        ".analysis_pipeline",
        "nlpo_toolkit.corpus_analysis.token_artifact",
        "nlpo_toolkit.corpus_analysis.diagnostic_trace",
        "nlpo_toolkit.corpus_analysis.analysis_cache",
        "nlpo_toolkit.corpus_analysis.runner",
        "nlpo_toolkit.corpus_analysis.analysis_pipeline",
    }
    assert modules.isdisjoint(forbidden)


def test_importing_analysis_records_does_not_load_artifact_modules() -> None:
    code = """
import sys
import nlpo_toolkit.corpus_analysis.analysis_records

assert "nlpo_toolkit.corpus_analysis.token_artifact" not in sys.modules
assert "nlpo_toolkit.corpus_analysis.diagnostic_trace" not in sys.modules
assert "nlpo_toolkit.corpus_analysis.analysis_cache" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_token_artifact_output_schema_and_row_are_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "tokens.tsv"
    record = TokenRecord(
        group="text",
        source_file="input/text.txt",
        section=None,
        chunk_index=0,
        sentence_index=0,
        token_index=0,
        global_token_index=0,
        char_start_in_chunk=0,
        char_end_in_chunk=4,
        char_start_in_text=0,
        char_end_in_text=4,
        sentence="Arma virumque.",
        token="Arma",
        lemma="arma",
        upos="NOUN",
        analysis_key="arma",
        included=True,
        exclusion_reason=None,
        ref_tag=None,
    )

    with TokenArtifactWriter(path, metadata=TokenArtifactMetadata(group="text")) as writer:
        writer.write(record)

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))

    assert rows[0] == list(TOKEN_ARTIFACT_COLUMNS)
    assert rows[1] == [
        "text",
        "input/text.txt",
        "",
        "0",
        "0",
        "0",
        "0",
        "0",
        "4",
        "0",
        "4",
        "Arma virumque.",
        "Arma",
        "arma",
        "NOUN",
        "arma",
        "true",
        "",
        "",
    ]
