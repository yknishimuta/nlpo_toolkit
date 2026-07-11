from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

import yaml

from .comparison import ComparisonSpec
from .partition_validation import PartitionSpec


AnalysisUnit = Literal["lemma", "surface"]
GroupingMode = Literal["groups", "per_file", "auto_single_cleaned"]
NLPBackend = Literal["stanza", "transformers"]

KNOWN_TOP_LEVEL_KEYS = frozenset(
    {
        "analysis_cache",
        "analysis_unit",
        "archive",
        "artifacts",
        "comparisons",
        "csv_header",
        "dictcheck",
        "filters",
        "grouping",
        "groups",
        "nlp",
        "normalization",
        "out_dir",
        "preprocess",
        "prune",
        "ref_tags",
        "trace",
        "validations",
    }
)

KNOWN_PREPROCESS_KEYS = frozenset({"kind", "config"})
KNOWN_GROUPING_KEYS = frozenset({"mode", "auto_group_name"})
KNOWN_GROUP_KEYS = frozenset({"files"})
KNOWN_NLP_KEYS = frozenset(
    {"backend", "language", "stanza_package", "model_name", "cpu_only"}
)
KNOWN_FILTER_KEYS = frozenset(
    {
        "drop_roman_numerals",
        "min_token_length",
        "roman_exceptions_file",
        "upos_targets",
    }
)
KNOWN_NORMALIZATION_KEYS = frozenset(
    {
        "casefold",
        "enabled",
        "map_i_j",
        "map_u_v",
        "normalize_ligatures",
        "strip_diacritics",
        "unicode_nf",
    }
)
KNOWN_DICTCHECK_KEYS = frozenset({"enabled", "wordlist", "lemma_normalize"})
KNOWN_REF_TAGS_KEYS = frozenset({"enabled", "patterns"})
KNOWN_TRACE_KEYS = frozenset(
    {"enabled", "path", "max_rows", "only_keys", "write_truncation_marker"}
)
KNOWN_ARTIFACTS_KEYS = frozenset({"tokens"})
KNOWN_TOKEN_ARTIFACT_KEYS = frozenset({"enabled", "path"})
KNOWN_ARCHIVE_KEYS = frozenset(
    {"enabled", "runs_dir", "include_input", "include_cleaned"}
)
KNOWN_ANALYSIS_CACHE_KEYS = frozenset(
    {"enabled", "dir", "use_manifest", "manifest_key_mode", "lock_timeout_sec"}
)
KNOWN_PRUNE_KEYS = frozenset({"keep_days", "keep_files", "lock_ttl_sec"})
KNOWN_VALIDATIONS_KEYS = frozenset({"partitions"})


@dataclass(frozen=True)
class GroupConfig:
    files: tuple[str, ...]


@dataclass(frozen=True)
class PreprocessConfig:
    kind: str | None = None
    config: str | None = None


@dataclass(frozen=True)
class GroupingConfig:
    mode: GroupingMode = "groups"
    auto_group_name: str = "text"


@dataclass(frozen=True)
class NLPConfig:
    backend: NLPBackend = "stanza"
    language: str = "la"
    stanza_package: str | dict[str, str] | None = "perseus"
    model_name: str | None = None
    cpu_only: bool = True


@dataclass(frozen=True)
class FilterConfig:
    min_token_length: int = 0
    drop_roman_numerals: bool = False
    roman_exceptions_file: str | None = None
    upos_targets: frozenset[str] = frozenset({"NOUN"})


@dataclass(frozen=True)
class NormalizationConfig:
    enabled: bool = True
    casefold: bool = False
    map_u_v: bool = False
    map_i_j: bool = False
    strip_diacritics: bool = False
    normalize_ligatures: bool = False
    unicode_nf: str | None = None


@dataclass(frozen=True)
class DictCheckConfig:
    enabled: bool = False
    wordlist: str | None = None
    lemma_normalize: str | None = None


@dataclass(frozen=True)
class RefTagsConfig:
    enabled: bool = False
    patterns: str | None = None


@dataclass(frozen=True)
class TraceConfig:
    enabled: bool = False
    path: str | None = None
    max_rows: int | None = 0
    only_keys: frozenset[str] = frozenset()
    write_truncation_marker: bool = True


@dataclass(frozen=True)
class TokenArtifactConfig:
    enabled: bool = False
    path: str = "output/tokens.tsv"


@dataclass(frozen=True)
class ArtifactsConfig:
    tokens: TokenArtifactConfig = field(default_factory=TokenArtifactConfig)


@dataclass(frozen=True)
class ArchiveConfig:
    enabled: bool = False
    runs_dir: str = "runs"
    include_input: bool = False
    include_cleaned: bool = False


@dataclass(frozen=True)
class AnalysisCacheConfig:
    enabled: bool = False
    directory: str = ".analysis_cache"
    use_manifest: bool = True
    manifest_key_mode: str = "relative"
    lock_timeout_sec: float = 300.0


@dataclass(frozen=True)
class PruneConfig:
    keep_days: int | None = None
    keep_files: int | None = None
    lock_ttl_sec: int | None = None


@dataclass(frozen=True)
class AppConfig:
    groups: Mapping[str, GroupConfig]
    preprocess: PreprocessConfig = PreprocessConfig()
    grouping: GroupingConfig = GroupingConfig()
    nlp: NLPConfig = NLPConfig()
    filters: FilterConfig = FilterConfig()
    normalization: NormalizationConfig = NormalizationConfig()
    dictcheck: DictCheckConfig = DictCheckConfig()
    ref_tags: RefTagsConfig = RefTagsConfig()
    trace: TraceConfig = TraceConfig()
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    archive: ArchiveConfig = ArchiveConfig()
    analysis_cache: AnalysisCacheConfig = AnalysisCacheConfig()
    prune: PruneConfig = PruneConfig()
    analysis_unit: AnalysisUnit = "lemma"
    out_dir: str = "output"
    csv_header: tuple[str, str] | None = None
    comparisons: tuple[ComparisonSpec, ...] = ()
    partition_validations: tuple[PartitionSpec, ...] = ()


def _reject_unknown_keys(
    mapping: Mapping[str, object],
    *,
    allowed: frozenset[str],
    context: str,
) -> None:
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if not unknown:
        return
    joined = ", ".join(unknown)
    raise ValueError(f"Unknown {context} key(s): {joined}")


def _as_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _optional_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if value is None:
        return {}
    return _as_mapping(value, context=context)


def _str_value(value: object, *, context: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _optional_str(value: object, *, context: str) -> str | None:
    if value is None:
        return None
    return _str_value(value, context=context)


def _bool_value(value: object, *, context: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be bool")
    return value


def _int_value(
    value: object,
    *,
    context: str,
    default: int | None = None,
    minimum: int | None = None,
) -> int:
    if value is None:
        if default is None:
            raise ValueError(f"{context} is required")
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{context} must be an integer greater than or equal to {minimum}")
    return value


def _optional_int(
    value: object,
    *,
    context: str,
    default: int | None = None,
    minimum: int | None = None,
) -> int | None:
    if value is None:
        return default
    return _int_value(value, context=context, minimum=minimum)


def _float_value(value: object, *, context: str, default: float, minimum_exclusive: float | None = None) -> float:
    if value is None:
        return default
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{context} must be a number")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{context} must be a finite number")
    if minimum_exclusive is not None and out <= minimum_exclusive:
        raise ValueError(f"{context} must be greater than {minimum_exclusive}")
    return out


def _string_tuple(value: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} must be a list of strings")
    out: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{context}[{index}] must be a string")
        out.append(item)
    return tuple(out)


def _optional_string_set(value: object, *, context: str, default: frozenset[str]) -> frozenset[str]:
    if value is None:
        return default
    if isinstance(value, str):
        raise ValueError(f"{context} must be a list of strings")
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of strings")
    out: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{context}[{index}] must be a string")
        stripped = item.strip()
        if stripped:
            out.add(stripped.upper())
    return frozenset(out)


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
    backend: NLPBackend = "transformers" if backend_raw == "transformers" else "stanza"

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


def _parse_prune_config(value: object) -> PruneConfig:
    prune = _optional_mapping(value, context="prune")
    _reject_unknown_keys(prune, allowed=KNOWN_PRUNE_KEYS, context="prune")
    return PruneConfig(
        keep_days=_optional_int(prune.get("keep_days"), context="prune.keep_days", minimum=0),
        keep_files=_optional_int(prune.get("keep_files"), context="prune.keep_files", minimum=0),
        lock_ttl_sec=_optional_int(prune.get("lock_ttl_sec"), context="prune.lock_ttl_sec", minimum=0),
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
    from .partition_validation import parse_partition_specs

    return tuple(parse_partition_specs(raw))


def _parse_comparison_specs(raw: Mapping[str, object]) -> tuple[ComparisonSpec, ...]:
    from .comparison import parse_comparison_specs

    return tuple(parse_comparison_specs(raw))


def _build_app_config(raw: Mapping[str, object]) -> AppConfig:
    _reject_unknown_keys(raw, allowed=KNOWN_TOP_LEVEL_KEYS, context="top-level config")
    if "groups" not in raw:
        raise ValueError("groups is required")

    validations = _optional_mapping(raw.get("validations"), context="validations")
    _reject_unknown_keys(validations, allowed=KNOWN_VALIDATIONS_KEYS, context="validations")

    groups = _parse_groups(raw["groups"])
    grouping = _parse_grouping_config(raw.get("grouping"))
    partition_validations = _parse_partition_specs(raw)
    if partition_validations and grouping.mode == "per_file":
        raise ValueError("validations.partitions cannot be used with grouping.mode: per_file")
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
        prune=_parse_prune_config(raw.get("prune")),
        analysis_unit=_parse_analysis_unit(raw.get("analysis_unit")),
        out_dir=_str_value(raw.get("out_dir", "output"), context="out_dir"),
        csv_header=_parse_csv_header(raw.get("csv_header")),
        comparisons=comparisons,
        partition_validations=partition_validations,
    )


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if path.suffix.lower() not in {".yml", ".yaml"}:
        raise ValueError("Config file must be YAML (.yml / .yaml)")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError("Top-level YAML must be a mapping.")
    return _build_app_config(data)


def ensure_app_config(config: AppConfig | Mapping[str, object]) -> AppConfig:
    if isinstance(config, AppConfig):
        return config
    return _build_app_config(config)


def _nlp_to_dict(config: NLPConfig) -> dict[str, object]:
    stanza_package: object
    if isinstance(config.stanza_package, Mapping):
        stanza_package = dict(config.stanza_package)
    else:
        stanza_package = config.stanza_package
    return {
        "backend": config.backend,
        "language": config.language,
        "stanza_package": stanza_package,
        "model_name": config.model_name,
        "cpu_only": config.cpu_only,
    }


def _normalization_to_dict(config: NormalizationConfig) -> dict[str, object]:
    return {
        "enabled": config.enabled,
        "casefold": config.casefold,
        "map_u_v": config.map_u_v,
        "map_i_j": config.map_i_j,
        "strip_diacritics": config.strip_diacritics,
        "normalize_ligatures": config.normalize_ligatures,
        "unicode_nf": config.unicode_nf,
    }


def _analysis_cache_to_dict(config: AnalysisCacheConfig) -> dict[str, object]:
    return {
        "enabled": config.enabled,
        "dir": config.directory,
        "use_manifest": config.use_manifest,
        "manifest_key_mode": config.manifest_key_mode,
        "lock_timeout_sec": config.lock_timeout_sec,
    }


def _preprocess_to_dict(config: PreprocessConfig) -> dict[str, object] | None:
    if config == PreprocessConfig():
        return None
    if config.kind != "cleaner":
        raise ValueError(f"Unsupported preprocess.kind for serialization: {config.kind!r}")
    return {
        "kind": "cleaner",
        "config": config.config,
    }


def _comparison_to_dict(spec: ComparisonSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "group_a": spec.group_a,
        "group_b": spec.group_b,
        "scale": spec.scale,
        "zero_correction": spec.zero_correction,
        "min_total_count": spec.min_total_count,
        "report": spec.report,
        "sort": {
            "by": spec.sort_by,
            "descending": spec.sort_descending,
        },
    }


def _partition_to_dict(spec: PartitionSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "whole": spec.whole,
        "parts": list(spec.parts),
        "on_mismatch": spec.on_mismatch,
        "report": spec.report,
    }


def config_to_dict(config: AppConfig) -> dict[str, object]:
    groups = {
        name: {"files": list(group.files)}
        for name, group in config.groups.items()
    }
    out: dict[str, object] = {
        "groups": groups,
        "grouping": {
            "mode": config.grouping.mode,
            "auto_group_name": config.grouping.auto_group_name,
        },
        "nlp": _nlp_to_dict(config.nlp),
        "filters": {
            "upos_targets": sorted(config.filters.upos_targets),
            "min_token_length": config.filters.min_token_length,
            "drop_roman_numerals": config.filters.drop_roman_numerals,
            "roman_exceptions_file": config.filters.roman_exceptions_file,
        },
        "normalization": _normalization_to_dict(config.normalization),
        "dictcheck": {
            "enabled": config.dictcheck.enabled,
            "wordlist": config.dictcheck.wordlist,
            "lemma_normalize": config.dictcheck.lemma_normalize,
        },
        "ref_tags": {
            "enabled": config.ref_tags.enabled,
            "patterns": config.ref_tags.patterns,
        },
        "trace": {
            "enabled": config.trace.enabled,
            "path": config.trace.path,
            "max_rows": config.trace.max_rows,
            "only_keys": sorted(config.trace.only_keys),
            "write_truncation_marker": config.trace.write_truncation_marker,
        },
        "artifacts": {
            "tokens": {
                "enabled": config.artifacts.tokens.enabled,
                "path": config.artifacts.tokens.path,
            },
        },
        "archive": {
            "enabled": config.archive.enabled,
            "runs_dir": config.archive.runs_dir,
            "include_input": config.archive.include_input,
            "include_cleaned": config.archive.include_cleaned,
        },
        "prune": {
            "keep_days": config.prune.keep_days,
            "keep_files": config.prune.keep_files,
            "lock_ttl_sec": config.prune.lock_ttl_sec,
        },
        "analysis_unit": config.analysis_unit,
        "out_dir": config.out_dir,
        "comparisons": [_comparison_to_dict(spec) for spec in config.comparisons],
        "validations": {
            "partitions": [_partition_to_dict(spec) for spec in config.partition_validations],
        },
    }
    preprocess = _preprocess_to_dict(config.preprocess)
    if preprocess is not None:
        out["preprocess"] = preprocess
    out["analysis_cache"] = _analysis_cache_to_dict(config.analysis_cache)
    if config.csv_header is not None:
        out["csv_header"] = list(config.csv_header)
    return out
