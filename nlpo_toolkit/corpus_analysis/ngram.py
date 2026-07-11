"""N-gram generation from complete token artifacts or configured text."""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, TextIO

from .config import load_config
from .corpus import prepare_corpora, resolve_corpus_work_items, run_preprocess_if_needed
from .analysis_records import TokenRecord
from .token_artifact import TokenArtifactError, read_token_records


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
    return tuple(
        row.get(column, "")
        for column in (
            "group",
            "source_file",
            "section",
            "chunk_index",
            "sentence_index",
        )
        if column in row
    ) or tuple(
        row.get(column, "")
        for column in ("group", "file", "label", "chunk", "sentence", "sent_idx")
        if column in row
    )


def _row_from_record(record: TokenRecord) -> dict[str, str]:
    return {
        "group": record.group,
        "source_file": record.source_file or "",
        "section": record.section or "",
        "chunk_index": str(record.chunk_index),
        "sentence_index": str(record.sentence_index),
        "token_index": str(record.token_index),
        "token": record.token,
        "lemma": record.lemma or "",
    }


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


def read_token_artifact_rows(
    tokens_path: Path,
    field: str,
    *,
    by_group: bool = False,
) -> list[dict[str, str]]:
    if field not in {"token", "lemma"}:
        raise NgramError("field must be 'token' or 'lemma'.")
    try:
        rows = [
            _row_from_record(record)
            for record in read_token_records(tokens_path, verify_hash=True)
            if record.included
        ]
    except TokenArtifactError as exc:
        raise NgramError(str(exc)) from exc
    return sorted(
        rows,
        key=lambda row: (
            row["group"],
            row["source_file"],
            row["section"],
            int(row["chunk_index"]),
            int(row["sentence_index"]),
            int(row["token_index"]),
        ),
    )


def _rows_from_text(text: str, group: str) -> list[dict[str, str]]:
    return [
        {"group": group, "token": match.group(0)}
        for match in _TOKEN_RE.finditer(text)
    ]


def read_config_text_rows(project_root: Path, config_path: Path, *, clean_mod: object | None = None) -> list[dict[str, str]]:
    cfg = load_config(config_path)
    rows: list[dict[str, str]] = []
    effective_clean_mod = clean_mod if clean_mod is not None else SimpleNamespace(main=lambda argv: 0)
    cleaned_dir = run_preprocess_if_needed(
        config=cfg,
        project_root=project_root,
        clean_mod=effective_clean_mod,
    )
    resolved = resolve_corpus_work_items(
        config=cfg,
        project_root=project_root,
        cleaned_dir=cleaned_dir,
    )
    for corpus in prepare_corpora(
        work_items=resolved.work_items,
        config=cfg,
        project_root=project_root,
    ):
        rows.extend(_rows_from_text(corpus.prepared_text, corpus.label))
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


def write_ngrams_from_tokens(
    *,
    tokens_path: Path,
    n: int,
    field: str,
    by_group: bool,
    min_count: int,
    top: int | None,
    output_format: str,
    out_path: Path | None,
) -> int:
    rows = build_ngrams_from_rows(
        read_token_artifact_rows(tokens_path, field, by_group=by_group),
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
    clean_mod: object | None = None,
) -> int:
    if field != "token":
        raise NgramError(
            "Config input currently supports --field token only. "
            "Use --tokens with a token artifact for lemma n-grams."
        )
    rows = build_ngrams_from_rows(
        read_config_text_rows(project_root, config_path, clean_mod=clean_mod),
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
