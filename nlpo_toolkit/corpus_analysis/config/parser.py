from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import yaml
from nlpo_toolkit.comparison.configured import ComparisonSpec
from ..partition_models import PartitionSpec

from .models import (
    AnalysisCacheConfig, AnalysisUnit, AppConfig, ArchiveConfig, ArtifactsConfig,
    DictCheckConfig, FilterConfig, GroupConfig, GroupingConfig, GroupingMode,
    NLPBackendName, NLPConfig, NormalizationConfig, PreprocessConfig,
    RefTagsConfig, TokenArtifactConfig, TraceConfig,
)
from .schema import (
    KNOWN_ANALYSIS_CACHE_KEYS, KNOWN_ARCHIVE_KEYS, KNOWN_ARTIFACTS_KEYS,
    KNOWN_DICTCHECK_KEYS, KNOWN_FILTER_KEYS, KNOWN_GROUP_KEYS,
    KNOWN_GROUPING_KEYS, KNOWN_NLP_KEYS, KNOWN_NORMALIZATION_KEYS,
    KNOWN_PREPROCESS_KEYS, KNOWN_REF_TAGS_KEYS, KNOWN_TOKEN_ARTIFACT_KEYS,
    KNOWN_TOP_LEVEL_KEYS, KNOWN_TRACE_KEYS, KNOWN_VALIDATIONS_KEYS,
    COMPARISON_KEYS, COMPARISON_SORT_KEYS, PARTITION_KEYS,
    COMPARISON_REPORT_VALUES, COMPARISON_SORT_BY_VALUES,
    reject_unknown_keys,
)
from .values import (
    as_mapping, bool_value, float_value, int_value, optional_int,
    optional_mapping, optional_str, optional_string_set, string_tuple, str_value,
)

_reject_unknown_keys = reject_unknown_keys
_as_mapping = as_mapping
_optional_mapping = optional_mapping
_str_value = str_value
_optional_str = optional_str
_bool_value = bool_value
_int_value = int_value
_optional_int = optional_int
_float_value = float_value
_string_tuple = string_tuple
_optional_string_set = optional_string_set


class ConfigError(ValueError):
    """A configuration file could not be read, parsed, or validated."""


def _parse_group_config(value: object, *, context: str) -> GroupConfig:
    group = _as_mapping(value, context=context)
    _reject_unknown_keys(group, allowed=KNOWN_GROUP_KEYS, context=context)
    if "files" not in group:
        raise ValueError(f"{context}.files is required")
    return GroupConfig(files=_string_tuple(group.get("files"), context=f"{context}.files"))


def _parse_groups(value: object) -> Mapping[str, GroupConfig]:
    groups = _as_mapping(value, context="groups")
    parsed: dict[str, GroupConfig] = {}
    for name, group_def in groups.items():
        if not isinstance(name, str) or not name:
            raise ValueError("groups keys must be non-empty strings")
        parsed[name] = _parse_group_config(group_def, context=f"groups.{name}")
    return parsed


def _parse_preprocess_config(value: object) -> PreprocessConfig:
    pp = _optional_mapping(value, context="preprocess")
    _reject_unknown_keys(pp, allowed=KNOWN_PREPROCESS_KEYS, context="preprocess")
    if not pp:
        return PreprocessConfig()
    kind = pp.get("kind")
    if kind is None:
        raise ValueError("preprocess.kind is required when preprocess is provided")
    if kind != "cleaner":
        raise ValueError(f"preprocess.kind must be 'cleaner' (got {kind!r})")
    return PreprocessConfig(
        kind="cleaner",
        config=_str_value(pp.get("config"), context="preprocess.config"),
    )


def _parse_grouping_config(value: object) -> GroupingConfig:
    grouping = _optional_mapping(value, context="grouping")
    _reject_unknown_keys(grouping, allowed=KNOWN_GROUPING_KEYS, context="grouping")
    raw_mode = grouping.get("mode", "groups")
    if raw_mode not in {"groups", "per_file", "auto_single_cleaned"}:
        raise ValueError("grouping.mode must be one of: groups, per_file, auto_single_cleaned")
    if raw_mode == "groups":
        mode: GroupingMode = "groups"
    elif raw_mode == "per_file":
        mode = "per_file"
    else:
        mode = "auto_single_cleaned"
    auto_group_name = grouping.get("auto_group_name", "text")
    return GroupingConfig(
        mode=mode,
        auto_group_name=_str_value(auto_group_name, context="grouping.auto_group_name"),
    )


def _parse_nlp_config(value: object) -> NLPConfig:
    nlp = _optional_mapping(value, context="nlp")
    _reject_unknown_keys(nlp, allowed=KNOWN_NLP_KEYS, context="nlp")
    backend_raw = nlp.get("backend", "stanza")
    if backend_raw not in {"stanza", "transformers"}:
        raise ValueError("nlp.backend must be one of: stanza, transformers")
    backend: NLPBackendName = "transformers" if backend_raw == "transformers" else "stanza"

    language_raw = nlp.get("language", "la")
    package_raw = nlp.get("stanza_package", "perseus")
    cpu_only_raw = nlp.get("cpu_only", True)

    if package_raw is not None and not isinstance(package_raw, (str, dict)):
        raise ValueError("nlp.stanza_package must be a string, mapping, or null")
    if package_raw is None:
        package: str | dict[str, str] | None = "perseus"
    elif isinstance(package_raw, dict):
        package: str | dict[str, str] | None = {}
        for key, value in package_raw.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("stanza_package mapping keys and values must be strings")
            package[key] = value
    else:
        package = package_raw

    model_name = _optional_str(nlp.get("model_name"), context="nlp.model_name")
    if backend == "transformers" and not model_name:
        raise ValueError("nlp.model_name is required when nlp.backend=transformers")

    return NLPConfig(
        backend=backend,
        language=_str_value(language_raw, context="nlp.language"),
        stanza_package=package,
        model_name=model_name,
        cpu_only=_bool_value(cpu_only_raw, context="nlp.cpu_only", default=True),
    )


def _parse_filter_config(value: object) -> FilterConfig:
    filters_map = _optional_mapping(value, context="filters")
    _reject_unknown_keys(filters_map, allowed=KNOWN_FILTER_KEYS, context="filters")
    return FilterConfig(
        min_token_length=_int_value(
            filters_map.get("min_token_length"),
            context="filters.min_token_length",
            default=0,
            minimum=0,
        ),
        drop_roman_numerals=_bool_value(
            filters_map.get("drop_roman_numerals"),
            context="filters.drop_roman_numerals",
            default=False,
        ),
        roman_exceptions_file=_optional_str(
            filters_map.get("roman_exceptions_file"),
            context="filters.roman_exceptions_file",
        ),
        upos_targets=_optional_string_set(
            filters_map.get("upos_targets"),
            context="filters.upos_targets",
            default=frozenset({"NOUN"}),
        ),
    )


def _parse_normalization_config(value: object) -> NormalizationConfig:
    norm = _optional_mapping(value, context="normalization")
    _reject_unknown_keys(norm, allowed=KNOWN_NORMALIZATION_KEYS, context="normalization")
    return NormalizationConfig(
        enabled=_bool_value(norm.get("enabled"), context="normalization.enabled", default=True),
        casefold=_bool_value(norm.get("casefold"), context="normalization.casefold", default=False),
        map_u_v=_bool_value(norm.get("map_u_v"), context="normalization.map_u_v", default=False),
        map_i_j=_bool_value(norm.get("map_i_j"), context="normalization.map_i_j", default=False),
        strip_diacritics=_bool_value(
            norm.get("strip_diacritics"),
            context="normalization.strip_diacritics",
            default=False,
        ),
        normalize_ligatures=_bool_value(
            norm.get("normalize_ligatures"),
            context="normalization.normalize_ligatures",
            default=False,
        ),
        unicode_nf=_optional_str(norm.get("unicode_nf"), context="normalization.unicode_nf"),
    )


def _parse_dictcheck_config(value: object) -> DictCheckConfig:
    dc = _optional_mapping(value, context="dictcheck")
    _reject_unknown_keys(dc, allowed=KNOWN_DICTCHECK_KEYS, context="dictcheck")
    return DictCheckConfig(
        enabled=_bool_value(dc.get("enabled"), context="dictcheck.enabled", default=False),
        wordlist=_optional_str(dc.get("wordlist"), context="dictcheck.wordlist"),
        lemma_normalize=_optional_str(dc.get("lemma_normalize"), context="dictcheck.lemma_normalize"),
    )


def _parse_ref_tags_config(value: object) -> RefTagsConfig:
    ref = _optional_mapping(value, context="ref_tags")
    _reject_unknown_keys(ref, allowed=KNOWN_REF_TAGS_KEYS, context="ref_tags")
    enabled = _bool_value(ref.get("enabled"), context="ref_tags.enabled", default=False)
    patterns = _optional_str(ref.get("patterns"), context="ref_tags.patterns")
    if enabled and not patterns:
        raise ValueError("ref_tags.patterns is required when ref_tags.enabled=true")
    return RefTagsConfig(
        enabled=enabled,
        patterns=patterns,
    )


def _parse_trace_config(value: object) -> TraceConfig:
    trace = _optional_mapping(value, context="trace")
    _reject_unknown_keys(trace, allowed=KNOWN_TRACE_KEYS, context="trace")
    only_keys = _optional_string_set(
        trace.get("only_keys"),
        context="trace.only_keys",
        default=frozenset(),
    )
    return TraceConfig(
        enabled=_bool_value(trace.get("enabled"), context="trace.enabled", default=False),
        path=_optional_str(trace.get("path"), context="trace.path"),
        max_rows=_optional_int(
            trace.get("max_rows"),
            context="trace.max_rows",
            default=0,
            minimum=0,
        ),
        only_keys=only_keys,
        write_truncation_marker=_bool_value(
            trace.get("write_truncation_marker"),
            context="trace.write_truncation_marker",
            default=True,
        ),
    )


def _parse_artifacts_config(value: object) -> ArtifactsConfig:
    artifacts = _optional_mapping(value, context="artifacts")
    _reject_unknown_keys(artifacts, allowed=KNOWN_ARTIFACTS_KEYS, context="artifacts")
    tokens = _optional_mapping(artifacts.get("tokens"), context="artifacts.tokens")
    _reject_unknown_keys(
        tokens,
        allowed=KNOWN_TOKEN_ARTIFACT_KEYS,
        context="artifacts.tokens",
    )
    return ArtifactsConfig(
        tokens=TokenArtifactConfig(
            enabled=_bool_value(
                tokens.get("enabled"),
                context="artifacts.tokens.enabled",
                default=False,
            ),
            path=_str_value(
                tokens.get("path", "output/tokens.tsv"),
                context="artifacts.tokens.path",
            ),
        )
    )


def _parse_archive_config(value: object) -> ArchiveConfig:
    archive = _optional_mapping(value, context="archive")
    _reject_unknown_keys(archive, allowed=KNOWN_ARCHIVE_KEYS, context="archive")
    return ArchiveConfig(
        enabled=_bool_value(archive.get("enabled"), context="archive.enabled", default=False),
        runs_dir=_str_value(archive.get("runs_dir", "runs"), context="archive.runs_dir"),
        include_input=_bool_value(
            archive.get("include_input"),
            context="archive.include_input",
            default=False,
        ),
        include_cleaned=_bool_value(
            archive.get("include_cleaned"),
            context="archive.include_cleaned",
            default=False,
        ),
    )


def _parse_analysis_cache_config(value: object) -> AnalysisCacheConfig:
    cache = _optional_mapping(value, context="analysis_cache")
    _reject_unknown_keys(cache, allowed=KNOWN_ANALYSIS_CACHE_KEYS, context="analysis_cache")
    return AnalysisCacheConfig(
        enabled=_bool_value(cache.get("enabled"), context="analysis_cache.enabled", default=False),
        directory=_str_value(
            cache.get("dir", ".analysis_cache"),
            context="analysis_cache.dir",
        ),
        use_manifest=_bool_value(
            cache.get("use_manifest"),
            context="analysis_cache.use_manifest",
            default=True,
        ),
        manifest_key_mode=_str_value(
            cache.get("manifest_key_mode", "relative"),
            context="analysis_cache.manifest_key_mode",
        ),
        lock_timeout_sec=_float_value(
            cache.get("lock_timeout_sec"),
            context="analysis_cache.lock_timeout_sec",
            default=300.0,
            minimum_exclusive=0.0,
        ),
    )


def _parse_analysis_unit(value: object) -> AnalysisUnit:
    unit = value if value is not None else "lemma"
    if unit not in {"lemma", "surface"}:
        raise ValueError("analysis_unit must be 'lemma' or 'surface'")
    return "surface" if unit == "surface" else "lemma"


def _parse_csv_header(value: object) -> tuple[str, str] | None:
    if value is None:
        return None
    values = _string_tuple(value, context="csv_header")
    if len(values) != 2 or not all(item.strip() for item in values):
        raise ValueError("csv_header must be a list[str] of length 2")
    return (values[0], values[1])


def _parse_partition_specs(raw: Mapping[str, object]) -> tuple[PartitionSpec, ...]:
    validations = raw.get("validations")
    if validations is None:
        return ()
    if not isinstance(validations, dict):
        raise ValueError("'validations' must be a mapping.")
    items = validations.get("partitions")
    if items is None:
        return ()
    if not isinstance(items, list):
        raise ValueError("'validations.partitions' must be a list.")
    group_names = set(_as_mapping(raw.get("groups"), context="groups"))
    specs: list[PartitionSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        label = f"validations.partitions[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be a mapping.")
        _reject_unknown_keys(item, allowed=PARTITION_KEYS, context=label)
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label}.name must be a non-empty string.")
        name = name.strip()
        if name in seen:
            raise ValueError(f"Duplicate partition name: {name}")
        seen.add(name)
        whole = item.get("whole")
        if not isinstance(whole, str) or not whole.strip():
            raise ValueError(f"{label}.whole must be a non-empty string.")
        whole = whole.strip()
        parts_raw = item.get("parts")
        if not isinstance(parts_raw, list) or not all(isinstance(part, str) and part.strip() for part in parts_raw):
            raise ValueError(f"{label}.parts must be list[str].")
        parts = tuple(part.strip() for part in parts_raw)
        if len(parts) < 2:
            raise ValueError(f"{label}.parts must contain at least 2 groups.")
        if len(set(parts)) != len(parts):
            raise ValueError(f"{label}.parts must not contain duplicate group names.")
        if whole in parts:
            raise ValueError(f"{label}.whole must not be included in parts.")
        on_mismatch = item.get("on_mismatch", "warn")
        if on_mismatch not in {"error", "warn"}:
            raise ValueError(f"{label}.on_mismatch must be 'warn' or 'error'.")
        report = item.get("report", "mismatches")
        if report not in {"mismatches", "all"}:
            raise ValueError(f"{label}.report must be 'mismatches' or 'all'.")
        missing = [group for group in (whole, *parts) if group not in group_names]
        if missing:
            raise ValueError(f"Partition {name} references unknown group(s): {', '.join(missing)}")
        specs.append(PartitionSpec(name, whole, parts, str(on_mismatch), str(report)))
    return tuple(specs)


def _parse_comparison_specs(raw: Mapping[str, object]) -> tuple[ComparisonSpec, ...]:
    items = raw.get("comparisons")
    if items is None:
        return ()
    if not isinstance(items, list):
        raise ValueError("'comparisons' must be a list.")
    group_names = set(_as_mapping(raw.get("groups"), context="groups"))
    specs: list[ComparisonSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        label = f"comparisons[{index}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{label} must be a mapping.")
        _reject_unknown_keys(item, allowed=COMPARISON_KEYS, context=label)
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label}.name must be a non-empty string.")
        name = name.strip()
        if name in seen:
            raise ValueError(f"Duplicate comparison name: {name}")
        seen.add(name)
        group_a = item.get("group_a")
        group_b = item.get("group_b")
        if not isinstance(group_a, str) or not group_a.strip():
            raise ValueError(f"comparison '{name}': group_a must be a non-empty string.")
        if not isinstance(group_b, str) or not group_b.strip():
            raise ValueError(f"comparison '{name}': group_b must be a non-empty string.")
        group_a, group_b = group_a.strip(), group_b.strip()
        if group_a == group_b:
            raise ValueError(f"comparison '{name}': group_a and group_b must be different.")
        if group_a not in group_names:
            raise ValueError(f"comparison '{name}': unknown group_a '{group_a}'")
        if group_b not in group_names:
            raise ValueError(f"comparison '{name}': unknown group_b '{group_b}'")
        scale = item.get("scale", 10_000)
        if not isinstance(scale, int) or isinstance(scale, bool) or scale <= 0:
            raise ValueError(f"comparison '{name}': scale must be a positive integer.")
        correction = item.get("zero_correction", 0.5)
        if not isinstance(correction, (int, float)) or isinstance(correction, bool) or not math.isfinite(float(correction)) or correction <= 0:
            raise ValueError(f"comparison '{name}': zero_correction must be a positive finite number.")
        minimum = item.get("min_total_count", 1)
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
            raise ValueError(f"comparison '{name}': min_total_count must be an integer >= 1.")
        report = item.get("report", "all")
        if report not in COMPARISON_REPORT_VALUES:
            raise ValueError(f"comparison '{name}': report must be 'all' or 'filtered'.")
        sort_by, descending = "log_likelihood", True
        sort = item.get("sort")
        if sort is not None:
            if not isinstance(sort, Mapping):
                raise ValueError(f"comparison '{name}': sort must be a mapping.")
            _reject_unknown_keys(sort, allowed=COMPARISON_SORT_KEYS, context=f"{label}.sort")
            sort_by = sort.get("by", "log_likelihood")
            if sort_by not in COMPARISON_SORT_BY_VALUES:
                raise ValueError(f"comparison '{name}': sort.by must be one of {sorted(COMPARISON_SORT_BY_VALUES)}.")
            descending = sort.get("descending", True)
            if not isinstance(descending, bool):
                raise ValueError(f"comparison '{name}': sort.descending must be bool.")
        specs.append(ComparisonSpec(name, group_a, group_b, int(scale), float(correction), int(minimum), str(report), str(sort_by), bool(descending)))
    return tuple(specs)


def parse_config(raw: Mapping[str, object]) -> AppConfig:
    _reject_unknown_keys(raw, allowed=KNOWN_TOP_LEVEL_KEYS, context="top-level config")
    if "groups" not in raw:
        raise ValueError("groups is required")

    validations = _optional_mapping(raw.get("validations"), context="validations")
    _reject_unknown_keys(validations, allowed=KNOWN_VALIDATIONS_KEYS, context="validations")

    groups = _parse_groups(raw["groups"])
    grouping = _parse_grouping_config(raw.get("grouping"))
    partition_validations = _parse_partition_specs(raw)
    comparisons = _parse_comparison_specs(raw)

    return AppConfig(
        groups=groups,
        preprocess=_parse_preprocess_config(raw.get("preprocess")),
        grouping=grouping,
        nlp=_parse_nlp_config(raw.get("nlp")),
        filters=_parse_filter_config(raw.get("filters")),
        normalization=_parse_normalization_config(raw.get("normalization")),
        dictcheck=_parse_dictcheck_config(raw.get("dictcheck")),
        ref_tags=_parse_ref_tags_config(raw.get("ref_tags")),
        trace=_parse_trace_config(raw.get("trace")),
        artifacts=_parse_artifacts_config(raw.get("artifacts")),
        archive=_parse_archive_config(raw.get("archive")),
        analysis_cache=_parse_analysis_cache_config(raw.get("analysis_cache")),
        analysis_unit=_parse_analysis_unit(raw.get("analysis_unit")),
        out_dir=_str_value(raw.get("out_dir", "output"), context="out_dir"),
        csv_header=_parse_csv_header(raw.get("csv_header")),
        comparisons=comparisons,
        partition_validations=partition_validations,
    )


def load_config(path: Path) -> AppConfig:
    if path.suffix.lower() not in {".yml", ".yaml"}:
        raise ConfigError("Config file must be YAML (.yml / .yaml)")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Config not found or unreadable: {path}: {exc}") from exc
    except UnicodeError as exc:
        raise ConfigError(f"Config file is not valid UTF-8: {path}: {exc}") from exc

    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in config file {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigError("Top-level YAML must be a mapping.")
    try:
        return parse_config(data)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def ensure_app_config(config: AppConfig | Mapping[str, object]) -> AppConfig:
    if isinstance(config, AppConfig):
        return config
    return parse_config(config)
