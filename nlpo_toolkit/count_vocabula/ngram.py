from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, TextIO

from .config import load_config
from .io_utils import expand_globs, read_concat
from .runner import _resolve_project_path


class NgramError(ValueError):
    pass


_HAS_WORD_CHAR_RE = re.compile(r"\w", re.UNICODE)
_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _open_output(path: Path | None) -> tuple[TextIO, bool]:
    if path is None:
        return sys.stdout, False
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8", newline=""), True


def _normalize_item(value: object) -> str | None:
    item = str(value or "").strip()
    if not item or not _HAS_WORD_CHAR_RE.search(item):
        return None
    if all(not char.isalnum() for char in item):
        return None
    return item.casefold()


def _sequence_key(row: dict[str, str], by_group: bool) -> tuple[str, ...]:
    parts: list[str] = []
    if "group" in row:
        parts.append(row.get("group", ""))
    for column in ("file", "label", "chunk"):
        if column in row:
            parts.append(row.get(column, ""))
    if "sentence" in row:
        parts.append(row.get("sentence", ""))
    elif "sent_idx" in row:
        parts.append(row.get("sent_idx", ""))
    return tuple(parts)


def _append_sequence_counts(
    counts_by_group: dict[str, Counter[str]],
    sequence: list[str],
    *,
    n: int,
    group: str,
) -> None:
    if len(sequence) < n:
        return
    counter = counts_by_group[group]
    for idx in range(0, len(sequence) - n + 1):
        counter[" ".join(sequence[idx:idx + n])] += 1


def _sorted_rows(
    counts_by_group: dict[str, Counter[str]],
    *,
    n: int,
    field: str,
    by_group: bool,
    min_count: int,
    top: int | None,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for group in sorted(counts_by_group):
        items = [
            (ngram, count)
            for ngram, count in counts_by_group[group].items()
            if count >= min_count
        ]
        items.sort(key=lambda item: (-item[1], item[0]))
        if top is not None:
            items = items[:top]
        for ngram, count in items:
            row: dict[str, str | int] = {
                "ngram": ngram,
                "count": count,
                "n": n,
                "field": field,
            }
            if by_group:
                row["group"] = group
            rows.append(row)
    return rows


def build_ngrams_from_rows(
    rows: Iterable[dict[str, str]],
    n: int,
    field: str,
    by_group: bool = False,
    min_count: int = 1,
    top: int | None = None,
) -> list[dict[str, str | int]]:
    if n < 1:
        raise NgramError("n must be 1 or greater.")
    if field not in {"token", "lemma"}:
        raise NgramError("field must be 'token' or 'lemma'.")
    if min_count < 1:
        raise NgramError("min-count must be 1 or greater.")
    if top is not None and top < 1:
        raise NgramError("top must be 1 or greater.")

    counts_by_group: dict[str, Counter[str]] = defaultdict(Counter)
    current_key: tuple[str, ...] | None = None
    current_group = ""
    sequence: list[str] = []

    for row in rows:
        key = _sequence_key(row, by_group)
        group = row.get("group", "") if by_group else ""
        if current_key is None:
            current_key = key
            current_group = group
        elif key != current_key:
            _append_sequence_counts(
                counts_by_group,
                sequence,
                n=n,
                group=current_group,
            )
            sequence = []
            current_key = key
            current_group = group

        item = _normalize_item(row.get(field, ""))
        if item is None:
            _append_sequence_counts(
                counts_by_group,
                sequence,
                n=n,
                group=current_group,
            )
            sequence = []
            continue
        sequence.append(item)

    if current_key is not None:
        _append_sequence_counts(
            counts_by_group,
            sequence,
            n=n,
            group=current_group,
        )

    return _sorted_rows(
        counts_by_group,
        n=n,
        field=field,
        by_group=by_group,
        min_count=min_count,
        top=top,
    )


def read_trace_rows(trace_path: Path, field: str, *, by_group: bool = False) -> list[dict[str, str]]:
    if not trace_path.exists():
        raise NgramError(f"Trace not found: {trace_path}")
    with trace_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        if field not in fieldnames:
            raise NgramError(f"Trace must contain '{field}' column.")
        if by_group and "group" not in fieldnames:
            raise NgramError("Trace must contain 'group' column when --by-group is used.")
        return list(reader)


def _rows_from_text(text: str, group: str) -> list[dict[str, str]]:
    return [
        {"group": group, "token": match.group(0)}
        for match in _TOKEN_RE.finditer(text)
    ]


def read_config_text_rows(project_root: Path, config_path: Path) -> list[dict[str, str]]:
    cfg = load_config(config_path)
    groups = cfg.get("groups") or {}
    rows: list[dict[str, str]] = []
    for group, group_def in groups.items():
        patterns = group_def.get("files") if isinstance(group_def, dict) else None
        if not isinstance(patterns, list):
            raise NgramError(f"groups.{group}.files must be list[str].")
        resolved_patterns = [
            str(_resolve_project_path(project_root, pattern))
            for pattern in patterns
        ]
        text = read_concat(expand_globs(resolved_patterns))
        rows.extend(_rows_from_text(text, str(group)))
    return rows


def write_ngram_rows(
    rows: list[dict[str, str | int]],
    *,
    output_format: str,
    out_path: Path | None,
    by_group: bool,
) -> int:
    if output_format not in {"tsv", "csv"}:
        raise NgramError("output format must be 'tsv' or 'csv'.")
    columns = ["ngram", "count", "n", "field"]
    if by_group:
        columns.append("group")
    delimiter = "\t" if output_format == "tsv" else ","
    out, should_close = _open_output(out_path)
    try:
        writer = csv.DictWriter(out, fieldnames=columns, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if should_close:
            out.close()
    return 0


def write_ngrams_from_trace(
    *,
    trace_path: Path,
    n: int,
    field: str,
    by_group: bool,
    min_count: int,
    top: int | None,
    output_format: str,
    out_path: Path | None,
) -> int:
    rows = build_ngrams_from_rows(
        read_trace_rows(trace_path, field, by_group=by_group),
        n=n,
        field=field,
        by_group=by_group,
        min_count=min_count,
        top=top,
    )
    return write_ngram_rows(
        rows,
        output_format=output_format,
        out_path=out_path,
        by_group=by_group,
    )


def write_ngrams_from_config(
    *,
    project_root: Path,
    config_path: Path,
    n: int,
    field: str,
    by_group: bool,
    min_count: int,
    top: int | None,
    output_format: str,
    out_path: Path | None,
) -> int:
    if field != "token":
        raise NgramError("Config input currently supports --field token only. Use --trace for lemma n-grams.")
    rows = build_ngrams_from_rows(
        read_config_text_rows(project_root, config_path),
        n=n,
        field=field,
        by_group=by_group,
        min_count=min_count,
        top=top,
    )
    return write_ngram_rows(
        rows,
        output_format=output_format,
        out_path=out_path,
        by_group=by_group,
    )
