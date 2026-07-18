"""NLP token analysis records and filter evaluation.

This module contains artifact-independent intermediate representations.
It must not depend on token artifact or diagnostic trace serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from nlpo_toolkit.nlp.contracts import (
    NLPBackend,
    NLPDocument,
    NLPSentence,
    UDMorphFeature,
)
from nlpo_toolkit.nlp.roman_numerals import (
    effective_roman_exceptions,
    should_drop_roman_numeral,
)
from .analysis_policy import (
    AnalysisExtractionPolicy,
    DEFAULT_ANALYSIS_EXTRACTION_POLICY,
    iter_analysis_chunks,
)

__all__ = [
    "AnalysisOptions",
    "NLPAnalysisRecord",
    "TokenRecord",
    "evaluate_analysis_record",
    "iter_nlp_analysis_records_from_text",
]


@dataclass(frozen=True)
class NLPAnalysisRecord:
    chunk_index: int
    sentence_index: int
    token_index: int
    global_token_index: int
    char_start_in_chunk: int | None
    char_end_in_chunk: int | None
    char_start_in_text: int | None
    char_end_in_text: int | None
    sentence: str
    token: str
    lemma: str | None
    upos: str | None
    morphology: tuple[UDMorphFeature, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "morphology", tuple(self.morphology))


@dataclass(frozen=True)
class TokenRecord:
    group: str
    source_file: str | None
    chunk_index: int
    sentence_index: int
    token_index: int
    global_token_index: int
    char_start_in_chunk: int | None
    char_end_in_chunk: int | None
    char_start_in_text: int | None
    char_end_in_text: int | None
    sentence: str
    token: str
    lemma: str | None
    upos: str | None
    analysis_key: str | None
    included: bool
    exclusion_reason: str | None
    ref_tag: str | None
    section: str | None = None
    morphology: tuple[UDMorphFeature, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "morphology", tuple(self.morphology))


@dataclass(frozen=True)
class AnalysisOptions:
    group: str
    source_files: tuple[Path, ...]
    use_lemma: bool
    upos_targets: frozenset[str]
    min_token_length: int = 0
    drop_roman_numerals: bool = False
    roman_exceptions: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_files", tuple(self.source_files))
        object.__setattr__(self, "upos_targets", frozenset(self.upos_targets))
        object.__setattr__(self, "roman_exceptions", frozenset(self.roman_exceptions))


def _sentence_text(sentence: NLPSentence) -> str:
    if sentence.text:
        return sentence.text
    return " ".join(token.text for token in sentence.tokens)


def _token_key_from_values(
    token: str, lemma: str | None, *, use_lemma: bool
) -> str | None:
    selected = lemma if (use_lemma and lemma) else token
    if selected is None:
        return None
    key = str(selected).strip().lower()
    return key or None


def iter_nlp_analysis_records(
    *,
    document: NLPDocument,
    chunk_index: int,
    chunk_start_in_text: int,
    global_token_start: int,
) -> Iterator[NLPAnalysisRecord]:
    global_index = global_token_start
    for sentence_index, sentence in enumerate(document.sentences):
        sentence_text = _sentence_text(sentence)
        for token_index, token in enumerate(sentence.tokens):
            yield NLPAnalysisRecord(
                chunk_index=chunk_index,
                sentence_index=sentence_index,
                token_index=token_index,
                global_token_index=global_index,
                char_start_in_chunk=token.start_char,
                char_end_in_chunk=token.end_char,
                char_start_in_text=(
                    chunk_start_in_text + token.start_char
                    if token.start_char is not None
                    else None
                ),
                char_end_in_text=(
                    chunk_start_in_text + token.end_char
                    if token.end_char is not None
                    else None
                ),
                sentence=sentence_text,
                token=token.text,
                lemma=token.lemma,
                upos=token.upos,
                morphology=token.morphology,
            )
            global_index += 1


def iter_nlp_analysis_records_from_text(
    *,
    text: str,
    nlp: NLPBackend,
    policy: AnalysisExtractionPolicy = DEFAULT_ANALYSIS_EXTRACTION_POLICY,
) -> Iterator[NLPAnalysisRecord]:
    if not text:
        return

    global_index = 0
    chunk_base_offset = 0

    for chunk_index, chunk in enumerate(iter_analysis_chunks(text, policy=policy)):
        doc = nlp(chunk)
        emitted = 0
        for record in iter_nlp_analysis_records(
            document=doc,
            chunk_index=chunk_index,
            chunk_start_in_text=chunk_base_offset,
            global_token_start=global_index,
        ):
            emitted += 1
            yield record
        global_index += emitted
        chunk_base_offset += len(chunk)


def evaluate_analysis_record(
    record: NLPAnalysisRecord,
    *,
    options: AnalysisOptions,
) -> TokenRecord:
    source_file = (
        str(options.source_files[0]) if len(options.source_files) == 1 else None
    )
    key = _token_key_from_values(
        record.token, record.lemma, use_lemma=options.use_lemma
    )
    effective_exceptions = effective_roman_exceptions(
        use_lemma=options.use_lemma,
        configured_exceptions=options.roman_exceptions,
    )
    exclusion_reason: str | None = None
    if key is None:
        exclusion_reason = "missing_key"
    elif record.upos not in options.upos_targets:
        exclusion_reason = "upos_not_targeted"
    elif len(key) < options.min_token_length:
        exclusion_reason = "too_short"
    elif should_drop_roman_numeral(
        key,
        drop_roman_numerals=options.drop_roman_numerals,
        effective_exceptions=effective_exceptions,
    ):
        exclusion_reason = "roman_numeral"

    return TokenRecord(
        group=options.group,
        source_file=source_file,
        section=None,
        chunk_index=record.chunk_index,
        sentence_index=record.sentence_index,
        token_index=record.token_index,
        global_token_index=record.global_token_index,
        char_start_in_chunk=record.char_start_in_chunk,
        char_end_in_chunk=record.char_end_in_chunk,
        char_start_in_text=record.char_start_in_text,
        char_end_in_text=record.char_end_in_text,
        sentence=record.sentence,
        token=record.token,
        lemma=record.lemma,
        upos=record.upos,
        analysis_key=key,
        included=exclusion_reason is None,
        exclusion_reason=exclusion_reason,
        ref_tag=None,
        morphology=record.morphology,
    )
