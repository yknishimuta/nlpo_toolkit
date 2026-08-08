from __future__ import annotations

import unicodedata

from ..analysis_records import NLPAnalysisRecord
from .character_text import normalize_character_stream
from .errors import FeatureError
from .filtering import is_word_token_text
from .models import AnalyzedFeatureCorpus, CharacterNgramMode


CONTENT_MASK_UPOS = frozenset({"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM", "X"})
CONTENT_MASK_POLICY_ID = "upos-content-v1"


def should_mask_content_record(record: NLPAnalysisRecord) -> bool:
    upos = (record.upos or "").strip().upper()
    return upos in CONTENT_MASK_UPOS or (not upos and is_word_token_text(record.token))


def _unit_text_and_absolute_start(corpus: AnalyzedFeatureCorpus) -> tuple[str, int]:
    if corpus.text is None:
        sample_id = corpus.sample.sample_id if corpus.sample else corpus.source.label
        raise FeatureError(
            "content-masked character n-grams require exact token offsets; "
            f"sample {sample_id!r} has no usable character span"
        )
    if corpus.sample is None:
        return corpus.text, 0
    if not corpus.raw_records or corpus.raw_records[0].char_start_in_text is None:
        raise FeatureError(
            "content-masked character n-grams require exact token offsets; "
            f"sample {corpus.sample.sample_id!r} has no usable character span"
        )
    return corpus.text, corpus.raw_records[0].char_start_in_text


def build_content_masked_text(corpus: AnalyzedFeatureCorpus) -> str:
    text, absolute_start = _unit_text_and_absolute_start(corpus)
    usable_upos = any(
        is_word_token_text(record.token) and bool((record.upos or "").strip())
        for record in corpus.raw_records
    )
    if corpus.sample is None and not usable_upos:
        raise FeatureError(
            "content-masked character n-grams require UPOS annotations; "
            f"no usable UPOS tags were found for corpus {corpus.source.label!r}"
        )
    masked = [False] * len(text)
    sample_id = corpus.sample.sample_id if corpus.sample else corpus.source.label
    source_length = len(corpus.source.prepared_text)
    for record in corpus.raw_records:
        if not should_mask_content_record(record):
            continue
        start = record.char_start_in_text
        end = record.char_end_in_text
        if (
            start is None
            or end is None
            or start < 0
            or start >= end
            or end > source_length
            or start < absolute_start
            or end > absolute_start + len(text)
        ):
            raise FeatureError(
                "content-masked character n-grams require exact token offsets; "
                f"record {record.global_token_index} in sample {sample_id!r} "
                "has no usable character span"
            )
        for index in range(start - absolute_start, end - absolute_start):
            if unicodedata.category(text[index]).startswith(("L", "M", "N")):
                masked[index] = True
    return "".join("*" if flag else character for character, flag in zip(text, masked))


def build_content_masked_character_stream(corpus: AnalyzedFeatureCorpus) -> str:
    return normalize_character_stream(
        build_content_masked_text(corpus), mode=CharacterNgramMode.FULL
    )
