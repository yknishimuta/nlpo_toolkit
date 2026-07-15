from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from nlpo_toolkit.nlp.roman_numerals import (
    RomanExceptionsError,
    effective_roman_exceptions,
    load_roman_exceptions,
    should_drop_roman_numeral,
)
from nlpo_toolkit.nlp.contracts import NLPBackend

from .analysis_records import (
    NLPAnalysisRecord,
    iter_nlp_analysis_records_from_text,
)
from .analysis_policy import AnalysisExtractionPolicy, DEFAULT_ANALYSIS_EXTRACTION_POLICY
from .ports import FeatureCommandDependencies
from .corpus import prepare_corpora
from .config_references import ConfigReferenceError
from .run_plan import build_analysis_plan
from .requests import CorpusPreparationRequest
from .runtime import build_nlp_runtime


class FeatureError(RuntimeError):
    pass


UPOS_FEATURES = (
    "NOUN",
    "VERB",
    "ADJ",
    "ADV",
    "PROPN",
    "PRON",
    "ADP",
    "AUX",
    "CCONJ",
    "SCONJ",
    "PART",
    "DET",
    "NUM",
)
CONTENT_UPOS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
FUNCTION_UPOS = {"PRON", "ADP", "AUX", "CCONJ", "SCONJ", "PART", "DET"}
_PUNCT_CHARS = set(string.punctuation + "“”‘’«»…—–-­")
_FEATURE_SAFE_RE = re.compile(r"[^0-9A-Za-z_]+")


def feature_token_value(record: NLPAnalysisRecord) -> str:
    return record.token.strip().lower()


def feature_lemma_value(record: NLPAnalysisRecord) -> str:
    selected = record.lemma or record.token
    return str(selected).strip().lower()


def feature_upos_value(record: NLPAnalysisRecord) -> str:
    return record.upos or "X"


def is_word_token_text(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return any(ch.isalnum() for ch in text) and not all(ch in _PUNCT_CHARS for ch in text)


def safe_feature_name(value: str) -> str:
    name = _FEATURE_SAFE_RE.sub("_", str(value).strip().lower()).strip("_")
    return name or "empty"


@dataclass(frozen=True)
class FeatureFilterPolicy:
    min_token_length: int = 0
    drop_roman_numerals: bool = False
    roman_exceptions: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if isinstance(self.min_token_length, bool) or not isinstance(
            self.min_token_length, int
        ):
            raise TypeError("min_token_length must be an integer")
        if self.min_token_length < 0:
            raise ValueError("min_token_length must be non-negative")
        if not isinstance(self.drop_roman_numerals, bool):
            raise TypeError("drop_roman_numerals must be a bool")
        normalized = frozenset(
            str(item).strip().lower()
            for item in self.roman_exceptions
            if str(item).strip()
        )
        object.__setattr__(self, "roman_exceptions", normalized)


def filter_feature_records(
    records: Iterable[NLPAnalysisRecord],
    *,
    policy: FeatureFilterPolicy,
) -> tuple[NLPAnalysisRecord, ...]:
    out: list[NLPAnalysisRecord] = []
    effective_exceptions = effective_roman_exceptions(
        use_lemma=False,
        configured_exceptions=policy.roman_exceptions,
    )
    for record in records:
        key = feature_token_value(record)
        if not is_word_token_text(key):
            continue
        if len(key) < policy.min_token_length:
            continue
        if should_drop_roman_numeral(
            key,
            drop_roman_numerals=policy.drop_roman_numerals,
            effective_exceptions=effective_exceptions,
        ):
            continue
        out.append(record)
    return tuple(out)


def compute_basic_features(
    records: Sequence[NLPAnalysisRecord],
    text: str,
    group: str,
    file_count: int,
    *,
    raw_token_count: int,
    sentence_count: int,
) -> dict[str, Any]:
    word_token_count = len(records)
    lemmas = [feature_lemma_value(r) for r in records]
    tokens = [feature_token_value(r) for r in records]
    lemma_counts = Counter(lemmas)
    lemma_type_count = len(lemma_counts)
    token_type_count = len(set(tokens))
    hapax_lemma_count = sum(1 for count in lemma_counts.values() if count == 1)
    token_lengths = [len(t) for t in tokens]

    return {
        "group": group,
        "file_count": file_count,
        "char_count": len(text),
        "sentence_count": sentence_count,
        "token_count": raw_token_count,
        "word_token_count": word_token_count,
        "lemma_type_count": lemma_type_count,
        "token_type_count": token_type_count,
        "hapax_lemma_count": hapax_lemma_count,
        "hapax_lemma_ratio": hapax_lemma_count / lemma_type_count if lemma_type_count else 0.0,
        "mean_sentence_length": word_token_count / sentence_count if sentence_count else 0.0,
        "mean_token_length": sum(token_lengths) / len(token_lengths) if token_lengths else 0.0,
        "type_token_ratio": token_type_count / word_token_count if word_token_count else 0.0,
        "lemma_type_token_ratio": lemma_type_count / word_token_count if word_token_count else 0.0,
    }


def compute_upos_features(records: Sequence[NLPAnalysisRecord]) -> dict[str, Any]:
    denom = len(records)
    counts = Counter(feature_upos_value(r) for r in records)
    features: dict[str, Any] = {}
    for upos in UPOS_FEATURES:
        count = counts.get(upos, 0)
        features[f"upos_{upos}_count"] = count
        features[f"upos_{upos}_ratio"] = count / denom if denom else 0.0

    content_count = sum(counts.get(upos, 0) for upos in CONTENT_UPOS)
    function_count = sum(counts.get(upos, 0) for upos in FUNCTION_UPOS)
    features["content_word_count"] = content_count
    features["content_word_ratio"] = content_count / denom if denom else 0.0
    features["function_word_count"] = function_count
    features["function_word_ratio"] = function_count / denom if denom else 0.0
    return features


def feature_field_value(record: NLPAnalysisRecord, field: str) -> str:
    return feature_lemma_value(record) if field == "lemma" else feature_token_value(record)


def select_mfw(
    all_group_records: Sequence[Sequence[NLPAnalysisRecord]],
    n: int,
    field: str,
) -> list[str]:
    if n <= 0:
        return []
    if field not in {"lemma", "token"}:
        raise FeatureError("--field must be 'lemma' or 'token'")
    counter: Counter[str] = Counter()
    for records in all_group_records:
        for record in records:
            value = feature_field_value(record, field)
            if value:
                counter[value] += 1
    return [term for term, _count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def compute_mfw_features(
    records: Sequence[NLPAnalysisRecord],
    mfw_terms: Sequence[str],
    field: str,
) -> dict[str, Any]:
    denom = len(records)
    counter: Counter[str] = Counter()
    for record in records:
        value = feature_field_value(record, field)
        if value:
            counter[value] += 1
    return {
        f"mfw_{safe_feature_name(term)}": counter.get(term, 0) / denom if denom else 0.0
        for term in mfw_terms
    }


@dataclass(frozen=True)
class FeatureOptions:
    field: str = "lemma"
    mfw: int = 0
    include_upos: bool = True
    include_basic: bool = True
    filter_policy: FeatureFilterPolicy = FeatureFilterPolicy()
    extraction_policy: AnalysisExtractionPolicy = DEFAULT_ANALYSIS_EXTRACTION_POLICY


@dataclass(frozen=True)
class FeatureRequest:
    corpus: CorpusPreparationRequest
    field: str = "lemma"
    mfw: int = 0
    include_upos: bool = True
    include_basic: bool = True


@dataclass(frozen=True)
class PreparedFeatureCorpus:
    group: str
    files: tuple[Path, ...]
    text: str
    raw_record_count: int
    sentence_count: int
    feature_records: tuple[NLPAnalysisRecord, ...]


@dataclass(frozen=True)
class FeatureCommandResult:
    rows: tuple[dict[str, Any], ...]


def build_feature_rows(
    groups_texts: list[tuple[str, list[Path], str]],
    nlp: NLPBackend,
    options: FeatureOptions,
) -> list[dict[str, Any]]:
    if options.mfw < 0:
        raise FeatureError("--mfw must be non-negative")
    if options.field not in {"lemma", "token"}:
        raise FeatureError("--field must be 'lemma' or 'token'")

    prepared: list[PreparedFeatureCorpus] = []
    for group, files, text in groups_texts:
        raw_records = tuple(
            iter_nlp_analysis_records_from_text(
                text=text,
                nlp=nlp,
                policy=options.extraction_policy,
            )
        )
        prepared.append(
            PreparedFeatureCorpus(
                group=group,
                files=tuple(files),
                text=text,
                raw_record_count=len(raw_records),
                sentence_count=len(
                    {(record.chunk_index, record.sentence_index) for record in raw_records}
                ),
                feature_records=filter_feature_records(
                    raw_records,
                    policy=options.filter_policy,
                ),
            )
        )

    mfw_terms = select_mfw(
        [corpus.feature_records for corpus in prepared], options.mfw, options.field
    )
    rows: list[dict[str, Any]] = []
    for corpus in prepared:
        group, files, text, records = (
            corpus.group,
            corpus.files,
            corpus.text,
            corpus.feature_records,
        )
        row: dict[str, Any] = {"group": group}
        if options.include_basic:
            row.update(
                compute_basic_features(
                    records,
                    text,
                    group,
                    len(files),
                    raw_token_count=corpus.raw_record_count,
                    sentence_count=corpus.sentence_count,
                )
            )
        if options.include_upos:
            row.update(compute_upos_features(records))
        if mfw_terms:
            row.update(compute_mfw_features(records, mfw_terms, options.field))
        rows.append(row)
    return rows


def execute_feature_command(
    request: FeatureRequest,
    *,
    dependencies: FeatureCommandDependencies,
) -> FeatureCommandResult:
    try:
        plan = build_analysis_plan(
            request.corpus,
            dependencies=dependencies.planning,
            preprocess_mode="execute",
        )
    except (ConfigReferenceError, FileNotFoundError) as exc:
        raise FeatureError(str(exc)) from exc
    config = plan.config

    roman_exceptions_path = plan.config_files.path("filters.roman_exceptions_file")
    try:
        configured_roman_exceptions = (
            load_roman_exceptions(roman_exceptions_path)
            if roman_exceptions_path is not None
            else frozenset()
        )
    except RomanExceptionsError as exc:
        raise FeatureError(str(exc)) from exc

    options = FeatureOptions(
        field=request.field,
        mfw=request.mfw,
        include_upos=request.include_upos,
        include_basic=request.include_basic,
        # UPOS targets belong to Count selection, not stylistic feature filtering.
        filter_policy=FeatureFilterPolicy(
            min_token_length=config.filters.min_token_length,
            drop_roman_numerals=config.filters.drop_roman_numerals,
            roman_exceptions=configured_roman_exceptions,
        ),
        extraction_policy=dependencies.analysis.extraction_policy,
    )

    groups_texts: list[tuple[str, list[Path], str]] = []
    for corpus in prepare_corpora(
        work_items=plan.work_items,
        config=config,
        config_files=plan.config_files,
    ):
        groups_texts.append((corpus.label, list(corpus.files), corpus.prepared_text))

    built_backend = build_nlp_runtime(
        config=config,
        backend_factory=dependencies.analysis.backend_factory,
    )

    return FeatureCommandResult(
        rows=tuple(build_feature_rows(groups_texts, built_backend.backend, options))
    )
