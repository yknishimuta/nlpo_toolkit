from __future__ import annotations

import csv
import re
import string
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TextIO

from .config import AppConfig, ensure_app_config, load_config
from .io_utils import read_concat
from .normalizer import normalize_text
from .preprocess import run_preprocess_if_needed
from .ref_tags import load_ref_tag_patterns, strip_and_count_ref_tags
from .runner import (
    _build_work_items,
    _resolve_auto_single_cleaned_group,
    _resolve_group_files,
    _resolve_project_path,
)


class FeatureError(RuntimeError):
    pass


@dataclass(frozen=True)
class TokenRecord:
    token: str
    lemma: str
    upos: str
    sentence_index: int


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
_ROMAN_RE = re.compile(r"^(m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3}))$", re.I)
_FEATURE_SAFE_RE = re.compile(r"[^0-9A-Za-z_]+")


def is_word_token(token: Any, word: Any = None) -> bool:
    text = str(getattr(word, "text", token) or "").strip()
    if not text:
        return False
    return any(ch.isalnum() for ch in text) and not all(ch in _PUNCT_CHARS for ch in text)


def safe_feature_name(value: str) -> str:
    name = _FEATURE_SAFE_RE.sub("_", str(value).strip().lower()).strip("_")
    return name or "empty"


def extract_token_records(doc) -> list[TokenRecord]:
    records: list[TokenRecord] = []
    for sent_idx, sent in enumerate(getattr(doc, "sentences", []) or []):
        words = getattr(sent, "tokens", None)
        if words is None:
            words = getattr(sent, "words", None)
        if words is None:
            words = []
        for token in words:
            # Common model stores tokens directly. Stanza tokens may contain words.
            nested_words = getattr(token, "words", None)
            iter_words = nested_words if nested_words else [token]
            for word in iter_words:
                text = str(getattr(word, "text", "") or "")
                lemma = str(getattr(word, "lemma", None) or text).strip().lower()
                upos = str(getattr(word, "upos", "X") or "X")
                records.append(
                    TokenRecord(
                        token=text,
                        lemma=lemma,
                        upos=upos,
                        sentence_index=sent_idx,
                    )
                )
    return records


def _filtered_word_records(
    records: Iterable[TokenRecord],
    *,
    min_token_length: int = 0,
    drop_roman_numerals: bool = False,
) -> list[TokenRecord]:
    out: list[TokenRecord] = []
    for record in records:
        if not is_word_token(record.token, record):
            continue
        key = record.token.strip().lower()
        if len(key) < min_token_length:
            continue
        if drop_roman_numerals and _ROMAN_RE.fullmatch(key):
            continue
        out.append(record)
    return out


def compute_basic_features(
    records: list[TokenRecord],
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
    sentence_count = len({r.sentence_index for r in records}) if records else 0
    word_token_count = len(word_records)
    lemmas = [r.lemma for r in word_records if r.lemma]
    tokens = [r.token.strip().lower() for r in word_records if r.token.strip()]
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


def compute_upos_features(records: list[TokenRecord]) -> dict[str, Any]:
    word_records = _filtered_word_records(records)
    denom = len(word_records)
    counts = Counter(r.upos for r in word_records)
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


def select_mfw(all_group_records: list[list[TokenRecord]], n: int, field: str) -> list[str]:
    if n <= 0:
        return []
    if field not in {"lemma", "token"}:
        raise FeatureError("--field must be 'lemma' or 'token'")
    counter: Counter[str] = Counter()
    for records in all_group_records:
        for record in _filtered_word_records(records):
            value = record.lemma if field == "lemma" else record.token.strip().lower()
            if value and is_word_token(value, record):
                counter[value] += 1
    return [term for term, _count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def compute_mfw_features(records: list[TokenRecord], mfw_terms: list[str], field: str) -> dict[str, Any]:
    word_records = _filtered_word_records(records)
    denom = len(word_records)
    counter: Counter[str] = Counter()
    for record in word_records:
        value = record.lemma if field == "lemma" else record.token.strip().lower()
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


def build_feature_rows(
    groups_texts: list[tuple[str, list[Path], str]],
    nlp,
    options: FeatureOptions,
) -> list[dict[str, Any]]:
    if options.mfw < 0:
        raise FeatureError("--mfw must be non-negative")
    if options.field not in {"lemma", "token"}:
        raise FeatureError("--field must be 'lemma' or 'token'")

    prepared: list[tuple[str, list[Path], str, list[TokenRecord]]] = []
    for group, files, text in groups_texts:
        doc = nlp(text)
        prepared.append((group, files, text, extract_token_records(doc)))

    mfw_terms = select_mfw([records for *_rest, records in prepared], options.mfw, options.field)
    rows: list[dict[str, Any]] = []
    for group, files, text, records in prepared:
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


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    return names or ["group"]


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return value


def write_feature_matrix(rows: list[dict[str, Any]], out: Path | TextIO | None, format: str) -> None:
    if format not in {"csv", "tsv"}:
        raise FeatureError("--format must be csv or tsv")
    delimiter = "," if format == "csv" else "\t"
    close = False
    if out is None:
        f = sys.stdout
    elif hasattr(out, "write"):
        f = out  # type: ignore[assignment]
    else:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        f = path.open("w", encoding="utf-8", newline="")
        close = True

    try:
        fieldnames = _fieldnames(rows)
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _format_value(row.get(k, "")) for k in fieldnames})
    finally:
        if close:
            f.close()


def _strip_ref_tags_if_configured(text: str, config: AppConfig, project_root: Path) -> str:
    if not config.ref_tags.enabled:
        return text
    ref_file = config.ref_tags.patterns
    if not ref_file:
        return text
    ref_path = _resolve_project_path(project_root, ref_file)
    if not ref_path.exists():
        return text
    stripped, _counter = strip_and_count_ref_tags(text, load_ref_tag_patterns(ref_path))
    return stripped


def run_features(
    *,
    project_root: Path,
    config_path: Path,
    out: Path | None = None,
    output_format: str = "csv",
    field: str = "lemma",
    mfw: int = 0,
    include_upos: bool = True,
    include_basic: bool = True,
    group_by_file: bool = False,
    auto_single_cleaned: bool = False,
    error_on_empty_group: bool = False,
    build_pipeline_fn: Callable[[str, str, bool], tuple[Any, str]],
    clean_mod: Any,
    load_config_fn: Callable[[Path], AppConfig | Mapping[str, object]] = load_config,
) -> int:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    if not config_path.exists():
        raise FeatureError(f"Config file not found: {config_path}")

    config = ensure_app_config(load_config_fn(config_path))
    cleaned_dir = run_preprocess_if_needed(cfg=config, project_root=project_root, clean_mod=clean_mod)
    grouping_mode = config.grouping.mode
    auto_mode = bool(auto_single_cleaned) or grouping_mode == "auto_single_cleaned"
    per_file = (bool(group_by_file) or grouping_mode == "per_file") and not auto_mode

    if auto_mode:
        group_files = _resolve_auto_single_cleaned_group(
            cleaned_dir=cleaned_dir,
            group_name=config.grouping.auto_group_name,
        )
    else:
        group_files = _resolve_group_files(
            groups=config.groups,
            project_root=project_root,
            cleaned_dir=cleaned_dir,
        )

    empty_groups = [name for name, files in group_files.items() if not files]
    if error_on_empty_group and empty_groups:
        raise FeatureError(f"No files matched for group(s): {', '.join(empty_groups)}")

    options = FeatureOptions(
        field=field,
        mfw=mfw,
        include_upos=include_upos,
        include_basic=include_basic,
        min_token_length=config.filters.min_token_length,
        drop_roman_numerals=config.filters.drop_roman_numerals,
    )

    language = config.nlp.language
    stanza_package = config.nlp.stanza_package
    cpu_only = config.nlp.cpu_only
    nlp, _package = build_pipeline_fn(language, stanza_package, cpu_only)

    groups_texts: list[tuple[str, list[Path], str]] = []
    for group, files in _build_work_items(group_files=group_files, group_by_file=per_file):
        text = read_concat(files)
        text = _strip_ref_tags_if_configured(text, config, project_root)
        text = normalize_text(text, config)
        groups_texts.append((group, files, text))

    rows = build_feature_rows(groups_texts, nlp, options)
    write_feature_matrix(rows, out=out, format=output_format)
    return 0
