from __future__ import annotations

import csv
from pathlib import Path

from nlpo_toolkit.corpus_analysis.analysis_records import TokenRecord
from nlpo_toolkit.corpus_analysis.diagnostic_trace import (
    DIAGNOSTIC_TRACE_COLUMNS,
    DiagnosticTraceWriter,
)


def _record(**overrides) -> TokenRecord:
    data = {
        "group": "text",
        "source_file": "input/text.txt",
        "section": None,
        "chunk_index": 0,
        "sentence_index": 0,
        "token_index": 0,
        "global_token_index": 0,
        "char_start_in_chunk": 0,
        "char_end_in_chunk": 4,
        "char_start_in_text": 0,
        "char_end_in_text": 4,
        "sentence": "Arma virumque.",
        "token": "Arma",
        "lemma": "arma",
        "upos": "NOUN",
        "analysis_key": "arma",
        "included": True,
        "exclusion_reason": None,
        "ref_tag": None,
    }
    data.update(overrides)
    return TokenRecord(**data)


def _rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.reader(f, delimiter="\t"))


def test_diagnostic_trace_columns_are_stable() -> None:
    assert DIAGNOSTIC_TRACE_COLUMNS == (
        "label",
        "chunk",
        "sent_idx",
        "token_idx",
        "token_char_start_in_chunk",
        "token_char_start_in_text",
        "sentence",
        "token",
        "lemma",
        "upos",
        "ref_tag",
        "global_row",
    )


def test_diagnostic_trace_writes_only_included_rows(tmp_path: Path) -> None:
    path = tmp_path / "trace.tsv"
    with DiagnosticTraceWriter(path) as writer:
        writer.consider(_record(token="Arma"))
        writer.consider(
            _record(
                token="amat",
                lemma="amo",
                analysis_key="amo",
                included=False,
                exclusion_reason="upos_not_targeted",
            )
        )

    rows = _rows(path)
    assert rows[0] == list(DIAGNOSTIC_TRACE_COLUMNS)
    assert [row[7] for row in rows[1:]] == ["Arma"]


def test_diagnostic_trace_only_keys_are_normalized(tmp_path: Path) -> None:
    path = tmp_path / "trace.tsv"
    with DiagnosticTraceWriter(path, only_keys={" ARMA "}) as writer:
        writer.consider(_record(token="Arma", analysis_key="arma"))
        writer.consider(_record(token="Virum", lemma="vir", analysis_key="vir"))

    assert [row[7] for row in _rows(path)[1:]] == ["Arma"]


def test_diagnostic_trace_max_rows_writes_single_marker(tmp_path: Path) -> None:
    path = tmp_path / "trace.tsv"
    with DiagnosticTraceWriter(path, max_rows=1) as writer:
        writer.consider(_record(token="Arma", analysis_key="arma"))
        writer.consider(_record(token="Virum", lemma="vir", analysis_key="vir"))
        writer.consider(_record(token="Puella", lemma="puella", analysis_key="puella"))

    rows = _rows(path)
    assert [row[7] for row in rows[1:]] == [
        "Arma",
        "(trace stopped; counting continues)",
    ]
    assert rows[2][9] == "TRACE_TRUNCATED"


def test_diagnostic_trace_can_disable_truncation_marker(tmp_path: Path) -> None:
    path = tmp_path / "trace.tsv"
    with DiagnosticTraceWriter(path, max_rows=1, write_truncation_marker=False) as writer:
        writer.consider(_record(token="Arma", analysis_key="arma"))
        writer.consider(_record(token="Virum", lemma="vir", analysis_key="vir"))

    assert [row[7] for row in _rows(path)[1:]] == ["Arma"]


def test_diagnostic_trace_max_rows_zero_is_unlimited(tmp_path: Path) -> None:
    path = tmp_path / "trace.tsv"
    with DiagnosticTraceWriter(path, max_rows=0) as writer:
        writer.consider(_record(token="Arma", analysis_key="arma"))
        writer.consider(_record(token="Virum", lemma="vir", analysis_key="vir"))

    assert [row[7] for row in _rows(path)[1:]] == ["Arma", "Virum"]


def test_diagnostic_trace_writes_missing_offsets_as_empty_fields(tmp_path: Path) -> None:
    path = tmp_path / "trace.tsv"
    with DiagnosticTraceWriter(path) as writer:
        writer.consider(
            _record(
                char_start_in_chunk=None,
                char_start_in_text=None,
            )
        )

    row = _rows(path)[1]
    assert row[4] == ""
    assert row[5] == ""
