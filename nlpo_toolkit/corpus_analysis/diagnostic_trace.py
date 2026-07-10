"""Legacy diagnostic trace TSV output.

Diagnostic traces may be filtered or truncated and are not stable,
complete token artifacts.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Collection, Iterator

from .analysis_records import TokenRecord

__all__ = [
    "DiagnosticTraceWriter",
    "LEGACY_TRACE_COLUMNS",
    "read_legacy_trace_records",
]

LEGACY_TRACE_COLUMNS = (
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
        self._writer.writerow(LEGACY_TRACE_COLUMNS)
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


def _legacy_int(value: str | None, default: int = 0) -> int:
    try:
        return int(value or "")
    except ValueError:
        return default


def read_legacy_trace_records(path: Path) -> Iterator[TokenRecord]:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Trace not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            token = row.get("token", "")
            if token == "(trace stopped; counting continues)" or row.get("upos") == "TRACE_TRUNCATED":
                continue
            group = row.get("group") or row.get("label") or ""
            key = row.get("lemma") or row.get("token") or ""
            source_file = row.get("source_file") or row.get("file") or None
            start_in_chunk = row.get("token_char_start_in_chunk") or row.get("char_start_in_chunk")
            start_in_text = row.get("token_char_start_in_text") or row.get("char_start_in_text")
            yield TokenRecord(
                group=group,
                source_file=source_file,
                section=None,
                chunk_index=_legacy_int(row.get("chunk") or row.get("chunk_index")),
                sentence_index=_legacy_int(row.get("sent_idx") or row.get("sentence_index")),
                token_index=_legacy_int(row.get("token_idx") or row.get("token_index")),
                global_token_index=_legacy_int(row.get("global_row") or row.get("global_token_index"), idx + 1),
                char_start_in_chunk=_legacy_int(start_in_chunk) if start_in_chunk not in (None, "") else None,
                char_end_in_chunk=None,
                char_start_in_text=_legacy_int(start_in_text) if start_in_text not in (None, "") else None,
                char_end_in_text=None,
                sentence=row.get("sentence", ""),
                token=token,
                lemma=row.get("lemma") or None,
                upos=row.get("upos") or None,
                analysis_key=key.strip().lower() or None,
                included=True,
                exclusion_reason=None,
                ref_tag=row.get("ref_tag") or None,
            )
