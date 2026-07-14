"""Optional diagnostic trace TSV output for human inspection.

Diagnostic traces may be filtered or truncated and must not be used as
reusable downstream analysis input.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Collection

from .analysis_records import TokenRecord

__all__ = [
    "DIAGNOSTIC_TRACE_COLUMNS",
    "DiagnosticTraceWriter",
]

DIAGNOSTIC_TRACE_COLUMNS = (
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


class DiagnosticTraceWriter:
    def __init__(
        self,
        path: Path,
        *,
        max_rows: int = 0,
        only_keys: Collection[str] | None = None,
        write_truncation_marker: bool = True,
    ) -> None:
        self.path = Path(path)
        self.max_rows = max_rows
        self.only_keys = {str(key).strip().lower() for key in (only_keys or ()) if str(key).strip()}
        self.write_truncation_marker = write_truncation_marker
        self._file: Any = None
        self._writer: csv.writer[Any] | None = None
        self._written = 0
        self._truncated = False

    def __enter__(self) -> "DiagnosticTraceWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._file, delimiter="\t")
        self._writer.writerow(DIAGNOSTIC_TRACE_COLUMNS)
        return self

    def consider(self, record: TokenRecord) -> None:
        if self._writer is None or self._truncated or not record.included:
            return
        key = record.analysis_key or ""
        if self.only_keys and key not in self.only_keys:
            return
        if self.max_rows > 0 and self._written >= self.max_rows:
            if self.write_truncation_marker:
                self._writer.writerow(
                    [
                        record.group,
                        record.chunk_index,
                        record.sentence_index,
                        record.token_index,
                        record.char_start_in_chunk if record.char_start_in_chunk is not None else "",
                        record.char_start_in_text if record.char_start_in_text is not None else "",
                        record.sentence,
                        "(trace stopped; counting continues)",
                        "",
                        "TRACE_TRUNCATED",
                        "",
                        self._written + 1,
                    ]
                )
            self._truncated = True
            return
        self._writer.writerow(
            [
                record.group,
                record.chunk_index,
                record.sentence_index,
                record.token_index,
                record.char_start_in_chunk if record.char_start_in_chunk is not None else "",
                record.char_start_in_text if record.char_start_in_text is not None else "",
                record.sentence,
                record.token,
                record.lemma or "",
                record.upos or "",
                record.ref_tag or "",
                self._written + 1,
            ]
        )
        self._written += 1

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._file is not None:
            self._file.close()
