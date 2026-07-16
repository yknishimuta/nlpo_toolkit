"""N-gram generation from complete token artifacts or configured text."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .config_references import ConfigReferenceError
from .corpus import PreparedCorpus
from .execution_session import prepare_analysis_corpus_session
from .ports import ConfigNgramDependencies
from .requests import CorpusPreparationRequest
from .token_artifact.errors import TokenArtifactError
from .token_artifact.reader import read_token_records
from .token_sequences.fields import TokenField, token_field_value
from .token_sequences.grouping import (
    TokenSequenceError,
    build_token_sequence_collection,
)
from .token_sequences.models import SequenceItem, TokenSequence


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


@dataclass(frozen=True)
class NgramRow:
    ngram: str
    count: int
    n: int
    field: TokenField
    group: str | None = None

    def as_mapping(self, *, by_group: bool) -> dict[str, str | int]:
        row: dict[str, str | int] = {
            "ngram": self.ngram, "count": self.count,
            "n": self.n, "field": self.field,
        }
        if by_group:
            row["group"] = self.group or ""
        return row


def _append_run(
    counter: Counter[str], run: list[str], *, n: int,
) -> None:
    for index in range(len(run) - n + 1):
        counter[" ".join(run[index:index + n])] += 1


def build_ngrams_from_sequences(
    sequences: Iterable[TokenSequence[SequenceItem]],
    *,
    n: int,
    field: TokenField,
    by_group: bool = False,
    min_count: int = 1,
    top: int | None = None,
) -> list[NgramRow]:
    if n < 1:
        raise NgramError("n must be 1 or greater.")
    if field not in {"token", "lemma"}:
        raise NgramError("field must be 'token' or 'lemma'.")
    if min_count < 1:
        raise NgramError("min-count must be 1 or greater.")
    if top is not None and top < 1:
        raise NgramError("top must be 1 or greater.")

    counts_by_group: dict[str, Counter[str]] = defaultdict(Counter)
    for sequence in sequences:
        destination = sequence.id.group if by_group else ""
        run: list[str] = []
        for item in sequence.items:
            if not item.included:
                continue
            normalized = _normalize_item(token_field_value(item, field))
            if normalized is None:
                _append_run(counts_by_group[destination], run, n=n)
                run = []
            else:
                run.append(normalized)
        _append_run(counts_by_group[destination], run, n=n)

    rows: list[NgramRow] = []
    for group in sorted(counts_by_group):
        values = [
            pair for pair in counts_by_group[group].items() if pair[1] >= min_count
        ]
        values.sort(key=lambda pair: (-pair[1], pair[0]))
        if top is not None:
            values = values[:top]
        rows.extend(
            NgramRow(value, count, n, field, group if by_group else None)
            for value, count in values
        )
    return rows


@dataclass(frozen=True)
class ConfigSequenceToken:
    group: str
    source_file: str | None
    section: str | None
    chunk_index: int
    sentence_index: int
    token_index: int
    global_token_index: int
    sentence: str
    token: str
    lemma: str | None
    included: bool


def iter_config_sequence_tokens(
    corpora: Iterable[PreparedCorpus],
) -> Iterator[ConfigSequenceToken]:
    global_index = 0
    for corpus in corpora:
        for token_index, match in enumerate(_TOKEN_RE.finditer(corpus.prepared_text)):
            yield ConfigSequenceToken(
                group=corpus.label, source_file=None, section=None,
                chunk_index=0, sentence_index=0, token_index=token_index,
                global_token_index=global_index, sentence=corpus.prepared_text,
                token=match.group(0), lemma=None, included=True,
            )
            global_index += 1


@dataclass(frozen=True)
class ConfigNgramRequest:
    corpus: CorpusPreparationRequest
    n: int
    by_group: bool
    min_count: int
    top: int | None


@dataclass(frozen=True)
class TokenNgramRequest:
    tokens_path: Path
    n: int
    field: TokenField
    by_group: bool
    min_count: int
    top: int | None


@dataclass(frozen=True)
class NgramCommandResult:
    rows: tuple[NgramRow, ...]
    by_group: bool


def execute_token_ngram_command(request: TokenNgramRequest) -> NgramCommandResult:
    try:
        collection = build_token_sequence_collection(
            read_token_records(request.tokens_path, verify_hash=True)
        )
    except (TokenArtifactError, TokenSequenceError) as exc:
        raise NgramError(str(exc)) from exc
    rows = build_ngrams_from_sequences(
        collection.sequences, n=request.n, field=request.field,
        by_group=request.by_group, min_count=request.min_count, top=request.top,
    )
    return NgramCommandResult(tuple(rows), request.by_group)


def execute_config_ngram_command(
    *, request: ConfigNgramRequest, dependencies: ConfigNgramDependencies,
) -> NgramCommandResult:
    try:
        session = prepare_analysis_corpus_session(
            request.corpus, dependencies=dependencies.corpus,
        )
        collection = build_token_sequence_collection(
            iter_config_sequence_tokens(session.corpora)
        )
    except (ConfigReferenceError, TokenSequenceError) as exc:
        raise NgramError(str(exc)) from exc
    rows = build_ngrams_from_sequences(
        collection.sequences, n=request.n, field="token",
        by_group=request.by_group, min_count=request.min_count, top=request.top,
    )
    return NgramCommandResult(tuple(rows), request.by_group)
