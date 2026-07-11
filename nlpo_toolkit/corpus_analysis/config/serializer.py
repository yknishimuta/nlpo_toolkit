from __future__ import annotations

from typing import Mapping

from nlpo_toolkit.comparison.configured import ComparisonSpec
from ..partition_models import PartitionSpec
from .models import AnalysisCacheConfig, AppConfig, NLPConfig, NormalizationConfig, PreprocessConfig


def _nlp_to_dict(config: NLPConfig) -> dict[str, object]:
    package: object = dict(config.stanza_package) if isinstance(config.stanza_package, Mapping) else config.stanza_package
    return {"backend": config.backend, "language": config.language, "stanza_package": package, "model_name": config.model_name, "cpu_only": config.cpu_only}


def _normalization_to_dict(config: NormalizationConfig) -> dict[str, object]:
    return {"enabled": config.enabled, "casefold": config.casefold, "map_u_v": config.map_u_v, "map_i_j": config.map_i_j, "strip_diacritics": config.strip_diacritics, "normalize_ligatures": config.normalize_ligatures, "unicode_nf": config.unicode_nf}


def _analysis_cache_to_dict(config: AnalysisCacheConfig) -> dict[str, object]:
    return {"enabled": config.enabled, "dir": config.directory, "use_manifest": config.use_manifest, "manifest_key_mode": config.manifest_key_mode, "lock_timeout_sec": config.lock_timeout_sec}


def _preprocess_to_dict(config: PreprocessConfig) -> dict[str, object] | None:
    if config == PreprocessConfig():
        return None
    if config.kind != "cleaner":
        raise ValueError(f"Unsupported preprocess.kind for serialization: {config.kind!r}")
    return {"kind": "cleaner", "config": config.config}


def _comparison_to_dict(spec: ComparisonSpec) -> dict[str, object]:
    return {"name": spec.name, "group_a": spec.group_a, "group_b": spec.group_b, "scale": spec.scale, "zero_correction": spec.zero_correction, "min_total_count": spec.min_total_count, "report": spec.report, "sort": {"by": spec.sort_by, "descending": spec.sort_descending}}


def _partition_to_dict(spec: PartitionSpec) -> dict[str, object]:
    return {"name": spec.name, "whole": spec.whole, "parts": list(spec.parts), "on_mismatch": spec.on_mismatch, "report": spec.report}


def config_to_dict(config: AppConfig) -> dict[str, object]:
    out: dict[str, object] = {
        "groups": {name: {"files": list(group.files)} for name, group in config.groups.items()},
        "grouping": {"mode": config.grouping.mode, "auto_group_name": config.grouping.auto_group_name},
        "nlp": _nlp_to_dict(config.nlp),
        "filters": {"upos_targets": sorted(config.filters.upos_targets), "min_token_length": config.filters.min_token_length, "drop_roman_numerals": config.filters.drop_roman_numerals, "roman_exceptions_file": config.filters.roman_exceptions_file},
        "normalization": _normalization_to_dict(config.normalization),
        "dictcheck": {"enabled": config.dictcheck.enabled, "wordlist": config.dictcheck.wordlist, "lemma_normalize": config.dictcheck.lemma_normalize},
        "ref_tags": {"enabled": config.ref_tags.enabled, "patterns": config.ref_tags.patterns},
        "trace": {"enabled": config.trace.enabled, "path": config.trace.path, "max_rows": config.trace.max_rows, "only_keys": sorted(config.trace.only_keys), "write_truncation_marker": config.trace.write_truncation_marker},
        "artifacts": {"tokens": {"enabled": config.artifacts.tokens.enabled, "path": config.artifacts.tokens.path}},
        "archive": {"enabled": config.archive.enabled, "runs_dir": config.archive.runs_dir, "include_input": config.archive.include_input, "include_cleaned": config.archive.include_cleaned},
        "analysis_unit": config.analysis_unit,
        "out_dir": config.out_dir,
        "comparisons": [_comparison_to_dict(spec) for spec in config.comparisons],
        "validations": {"partitions": [_partition_to_dict(spec) for spec in config.partition_validations]},
        "analysis_cache": _analysis_cache_to_dict(config.analysis_cache),
    }
    preprocess = _preprocess_to_dict(config.preprocess)
    if preprocess is not None:
        out["preprocess"] = preprocess
    if config.csv_header is not None:
        out["csv_header"] = list(config.csv_header)
    return out
