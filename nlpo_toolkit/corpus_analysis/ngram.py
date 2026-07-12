"""N-gram generation from complete token artifacts or configured text."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .corpus import PreparedCorpus, prepare_corpora
from .dependencies import ConfigNgramDependencies
from .run_plan import build_corpus_plan
from .analysis_records import TokenRecord
from .token_artifact import TokenArtifactError, read_token_records


class NgramError(ValueError):
    pass


_HAS_WORD_CHAR_RE = re.compile(r"\w", re.UNICODE)
_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


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


def iter_config_token_rows(
    corpora: Iterable[PreparedCorpus],
) -> Iterator[dict[str, str]]:
    for corpus in corpora:
        for match in _TOKEN_RE.finditer(corpus.prepared_text):
            yield {"group": corpus.label, "token": match.group(0)}


@dataclass(frozen=True)
class ConfigNgramRequest:
    project_root: Path
    config_path: Path
    n: int
    field: str
    by_group: bool
    min_count: int
    top: int | None
    group_by_file: bool = False
    auto_single_cleaned: bool = False
    error_on_empty_group: bool = False


@dataclass(frozen=True)
class TokenNgramRequest:
    tokens_path: Path
    n: int
    field: str
    by_group: bool
    min_count: int
    top: int | None


@dataclass(frozen=True)
class NgramCommandResult:
    rows: tuple[dict[str, str | int], ...]
    by_group: bool


def execute_token_ngram_command(request: TokenNgramRequest) -> NgramCommandResult:
    rows = build_ngrams_from_rows(
        read_token_artifact_rows(
            request.tokens_path, request.field, by_group=request.by_group
        ),
        n=request.n,
        field=request.field,
        by_group=request.by_group,
        min_count=request.min_count,
        top=request.top,
    )
    return NgramCommandResult(rows=tuple(rows), by_group=request.by_group)


def execute_config_ngram_command(
    *,
    request: ConfigNgramRequest,
    dependencies: ConfigNgramDependencies,
) -> NgramCommandResult:
    if request.field != "token":
        raise NgramError(
            "Config input supports --field token only. "
            "Use --tokens with a token artifact for lemma n-grams."
        )
    plan = build_corpus_plan(
        project_root=request.project_root,
        script_dir=None,
        config_path=request.config_path,
        group_by_file=request.group_by_file,
        auto_single_cleaned=request.auto_single_cleaned,
        error_on_empty_group=request.error_on_empty_group,
        dependencies=dependencies.planning,
        preprocess_mode="execute",
    )
    corpora = prepare_corpora(
        work_items=plan.work_items,
        config=plan.config,
        project_root=plan.project_root,
    )
    rows = build_ngrams_from_rows(
        iter_config_token_rows(corpora),
        n=request.n,
        field=request.field,
        by_group=request.by_group,
        min_count=request.min_count,
        top=request.top,
    )
    return NgramCommandResult(rows=tuple(rows), by_group=request.by_group)
