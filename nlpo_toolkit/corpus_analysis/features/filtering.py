from __future__ import annotations

import re
import string
from collections.abc import Iterable

from nlpo_toolkit.nlp.roman_numerals import (
    effective_roman_exceptions,
    should_drop_roman_numeral,
)

from ..analysis_records import NLPAnalysisRecord
from .models import FeatureField, FeatureFilterPolicy


_PUNCT_CHARS = set(string.punctuation + "“”‘’«»…—–-­")
_FEATURE_SAFE_RE = re.compile(r"[^0-9A-Za-z_]+")


def normalize_feature_value(value: str) -> str:
    return value.strip().lower()


def feature_token_value(record: NLPAnalysisRecord) -> str:
    return normalize_feature_value(record.token)


def feature_lemma_value(record: NLPAnalysisRecord) -> str:
    return normalize_feature_value(record.lemma or record.token)


def feature_field_value(record: NLPAnalysisRecord, field: FeatureField) -> str:
    return (
        feature_lemma_value(record) if field == "lemma" else feature_token_value(record)
    )


def is_word_token_text(value: str) -> bool:
    text = str(value or "").strip()
    return (
        bool(text)
        and any(ch.isalnum() for ch in text)
        and not all(ch in _PUNCT_CHARS for ch in text)
    )


def safe_feature_name(value: str) -> str:
    return _FEATURE_SAFE_RE.sub("_", str(value).strip().lower()).strip("_") or "empty"


def is_feature_record_eligible(
    record: NLPAnalysisRecord,
    *,
    policy: FeatureFilterPolicy,
) -> bool:
    key = feature_token_value(record)
    if not is_word_token_text(key) or len(key) < policy.min_token_length:
        return False
    return not should_drop_roman_numeral(
        key,
        drop_roman_numerals=policy.drop_roman_numerals,
        effective_exceptions=effective_roman_exceptions(
            use_lemma=False,
            configured_exceptions=policy.roman_exceptions,
        ),
    )


def filter_feature_records(
    records: Iterable[NLPAnalysisRecord],
    *,
    policy: FeatureFilterPolicy,
) -> tuple[NLPAnalysisRecord, ...]:
    return tuple(
        record
        for record in records
        if is_feature_record_eligible(record, policy=policy)
    )
