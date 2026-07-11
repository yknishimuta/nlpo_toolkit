from __future__ import annotations

from typing import Mapping


KNOWN_TOP_LEVEL_KEYS = frozenset({"analysis_cache", "analysis_unit", "archive", "artifacts", "comparisons", "csv_header", "dictcheck", "filters", "grouping", "groups", "nlp", "normalization", "out_dir", "preprocess", "ref_tags", "trace", "validations"})
KNOWN_PREPROCESS_KEYS = frozenset({"kind", "config"})
KNOWN_GROUPING_KEYS = frozenset({"mode", "auto_group_name"})
KNOWN_GROUP_KEYS = frozenset({"files"})
KNOWN_NLP_KEYS = frozenset({"backend", "language", "stanza_package", "model_name", "cpu_only"})
KNOWN_FILTER_KEYS = frozenset({"drop_roman_numerals", "min_token_length", "roman_exceptions_file", "upos_targets"})
KNOWN_NORMALIZATION_KEYS = frozenset({"casefold", "enabled", "map_i_j", "map_u_v", "normalize_ligatures", "strip_diacritics", "unicode_nf"})
KNOWN_DICTCHECK_KEYS = frozenset({"enabled", "wordlist", "lemma_normalize"})
KNOWN_REF_TAGS_KEYS = frozenset({"enabled", "patterns"})
KNOWN_TRACE_KEYS = frozenset({"enabled", "path", "max_rows", "only_keys", "write_truncation_marker"})
KNOWN_ARTIFACTS_KEYS = frozenset({"tokens"})
KNOWN_TOKEN_ARTIFACT_KEYS = frozenset({"enabled", "path"})
KNOWN_ARCHIVE_KEYS = frozenset({"enabled", "runs_dir", "include_input", "include_cleaned"})
KNOWN_ANALYSIS_CACHE_KEYS = frozenset({"enabled", "dir", "use_manifest", "manifest_key_mode", "lock_timeout_sec"})
KNOWN_VALIDATIONS_KEYS = frozenset({"partitions"})
COMPARISON_KEYS = frozenset({"name", "group_a", "group_b", "scale", "zero_correction", "min_total_count", "report", "sort"})
COMPARISON_SORT_KEYS = frozenset({"by", "descending"})
PARTITION_KEYS = frozenset({"name", "whole", "parts", "on_mismatch", "report"})
COMPARISON_REPORT_VALUES = frozenset({"all", "filtered"})
COMPARISON_SORT_BY_VALUES = frozenset({"log_likelihood", "abs_log_ratio", "total_count", "item"})


def reject_unknown_keys(mapping: Mapping[str, object], *, allowed: frozenset[str], context: str) -> None:
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if unknown:
        raise ValueError(f"Unknown {context} key(s): {', '.join(unknown)}")
