from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from nlpo_toolkit.nlp import should_drop_roman_numeral

from .analysis_records import (
    NLPAnalysisRecord,
    iter_nlp_analysis_records_from_text,
)
from .analysis_policy import AnalysisExtractionPolicy, DEFAULT_ANALYSIS_EXTRACTION_POLICY
from .dependencies import FeatureCommandDependencies
from .corpus import prepare_corpora
from .run_plan import build_corpus_plan
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


def _filtered_word_records(
    records: Iterable[NLPAnalysisRecord],
    *,
    min_token_length: int = 0,
    drop_roman_numerals: bool = False,
) -> list[NLPAnalysisRecord]:
    out: list[NLPAnalysisRecord] = []
    for record in records:
        key = feature_token_value(record)
        if not is_word_token_text(key):
            continue
        if len(key) < min_token_length:
            continue
        if should_drop_roman_numeral(
            key,
            drop_roman_numerals=drop_roman_numerals,
            effective_exceptions=frozenset(),
        ):
            continue
        out.append(record)
    return out


def compute_basic_features(
    records: Sequence[NLPAnalysisRecord],
    text: str,
    group: str,
    file_count: int,
    *,
    min_token_length: int = 0,
    drop_roman_numerals: bool = False,
) -> dict[str, Any]:
    word_records = _filtered_word_records(
        records,
        min_token_length=min_token_length,
        drop_roman_numerals=drop_roman_numerals,
    )
    sentence_count = len({(r.chunk_index, r.sentence_index) for r in records})
    word_token_count = len(word_records)
    lemmas = [feature_lemma_value(r) for r in word_records]
    tokens = [feature_token_value(r) for r in word_records]
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
        "token_count": len(records),
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
    word_records = _filtered_word_records(records)
    denom = len(word_records)
    counts = Counter(feature_upos_value(r) for r in word_records)
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
        for record in _filtered_word_records(records):
            value = feature_lemma_value(record) if field == "lemma" else feature_token_value(record)
            if value and is_word_token_text(value):
                counter[value] += 1
    return [term for term, _count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def compute_mfw_features(
    records: Sequence[NLPAnalysisRecord],
    mfw_terms: Sequence[str],
    field: str,
) -> dict[str, Any]:
    word_records = _filtered_word_records(records)
    denom = len(word_records)
    counter: Counter[str] = Counter()
    for record in word_records:
        value = feature_lemma_value(record) if field == "lemma" else feature_token_value(record)
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
    min_token_length: int = 0
    drop_roman_numerals: bool = False
    extraction_policy: AnalysisExtractionPolicy = DEFAULT_ANALYSIS_EXTRACTION_POLICY


@dataclass(frozen=True)
class FeatureRequest:
    project_root: Path
    config_path: Path
    field: str = "lemma"
    mfw: int = 0
    include_upos: bool = True
    include_basic: bool = True
    group_by_file: bool = False
    auto_single_cleaned: bool = False
    error_on_empty_group: bool = False


@dataclass(frozen=True)
class PreparedFeatureCorpus:
    group: str
    files: tuple[Path, ...]
    text: str
    records: tuple[NLPAnalysisRecord, ...]


@dataclass(frozen=True)
class FeatureCommandResult:
    rows: tuple[dict[str, Any], ...]


def build_feature_rows(
    groups_texts: list[tuple[str, list[Path], str]],
    nlp,
    options: FeatureOptions,
) -> list[dict[str, Any]]:
    if options.mfw < 0:
        raise FeatureError("--mfw must be non-negative")
    if options.field not in {"lemma", "token"}:
        raise FeatureError("--field must be 'lemma' or 'token'")

    prepared: list[PreparedFeatureCorpus] = []
    for group, files, text in groups_texts:
        prepared.append(
            PreparedFeatureCorpus(
                group=group,
                files=tuple(files),
                text=text,
                records=tuple(
                    iter_nlp_analysis_records_from_text(
                        text=text,
                        nlp=nlp,
                        policy=options.extraction_policy,
                    )
                ),
            )
        )

    mfw_terms = select_mfw([corpus.records for corpus in prepared], options.mfw, options.field)
    rows: list[dict[str, Any]] = []
    for corpus in prepared:
        group, files, text, records = (
            corpus.group,
            corpus.files,
            corpus.text,
            corpus.records,
        )
        row: dict[str, Any] = {"group": group}
        if options.include_basic:
            row.update(
                compute_basic_features(
                    records,
                    text,
                    group,
                    len(files),
                    min_token_length=options.min_token_length,
                    drop_roman_numerals=options.drop_roman_numerals,
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
    except FileNotFoundError as exc:
        raise FeatureError(str(exc)) from exc
    config = plan.config

    options = FeatureOptions(
        field=request.field,
        mfw=request.mfw,
        include_upos=request.include_upos,
        include_basic=request.include_basic,
        min_token_length=config.filters.min_token_length,
        drop_roman_numerals=config.filters.drop_roman_numerals,
        extraction_policy=dependencies.analysis.extraction_policy,
    )

    groups_texts: list[tuple[str, list[Path], str]] = []
    for corpus in prepare_corpora(
        work_items=plan.work_items,
        config=config,
        project_root=plan.project_root,
    ):
        groups_texts.append((corpus.label, list(corpus.files), corpus.prepared_text))

    nlp, _backend_info, _package = build_nlp_runtime(
        config=config,
        backend_factory=dependencies.analysis.backend_factory,
    )

    return FeatureCommandResult(rows=tuple(build_feature_rows(groups_texts, nlp, options)))
