from __future__ import annotations

import math
from dataclasses import asdict, dataclass
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
        "analysis_unit",
        "archive",
        "cleaner_config",
        "comparisons",
        "cpu_only",
        "csv_header",
        "dictcheck",
        "filter",
        "filters",
        "group",
        "grouping",
        "groups",
        "language",
        "lemma_cache",
        "nlp",
        "normalization",
        "out_dir",
        "preprocess",
        "prune",
        "ref_tags",
        "stanza_package",
        "stanza_pkg",
        "trace",
        "upos_targets",
        "validations",
        "vocab_path",
    }
)

KNOWN_FILTER_KEYS = frozenset(
    {
        "drop_roman_numerals",
        "min_token_length",
        "roman_exception_files",
        "roman_exceptions_file",
    }
)


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
    exclude_lemmas: str | None = None


@dataclass(frozen=True)
class NormalizationConfig:
    enabled: bool = True
    casefold: bool = False
    uv: str | None = None
    ij: str | None = None
    diacritics: str | None = None
    ligatures: Mapping[str, str] | None = None
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
class ArchiveConfig:
    enabled: bool = False
    runs_dir: str = "runs"
    include_input: bool = False
    include_cleaned: bool = False


@dataclass(frozen=True)
class LemmaCacheConfig:
    enabled: bool = False
    directory: str = ".lemma_cache"
    use_manifest: bool = True
    manifest_key_mode: str = "absolute"
    lock_timeout_sec: float = 300.0
    include_ref_tags_in_config_hash: bool = True


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
    archive: ArchiveConfig = ArchiveConfig()
    lemma_cache: LemmaCacheConfig = LemmaCacheConfig()
    prune: PruneConfig = PruneConfig()
    analysis_unit: AnalysisUnit = "lemma"
    out_dir: str = "output"
    csv_header: tuple[str, str] | None = None
    vocab_path: str | None = None
    comparisons: tuple[ComparisonSpec, ...] = ()
    partition_validations: tuple[PartitionSpec, ...] = ()


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


def _normalize_groups(raw: Mapping[str, object]) -> dict[str, object]:
    cfg = dict(raw)
    if "groups" in cfg and cfg["groups"] is not None:
        return cfg

    if "group" in cfg and cfg["group"]:
        group = _as_mapping(cfg["group"], context="group")
        name = group.get("name", "text")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("group.name must be a non-empty string")
        files = group.get("files")
        if not files:
            raise ValueError("'group.files' is required.")
        cfg["groups"] = {name: {"files": _string_tuple(files, context="group.files")}}
        return cfg

    grouping = cfg.get("grouping")
    if isinstance(grouping, Mapping) and grouping.get("mode") == "auto_single_cleaned":
        name = str(grouping.get("auto_group_name") or "text")
        cfg["groups"] = {name: {"files": []}}
        return cfg

    raise ValueError("Config must define 'groups' or 'group'.")


def _parse_group_config(value: object, *, context: str) -> GroupConfig:
    group = _as_mapping(value, context=context)
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


def _parse_nlp_config(raw: Mapping[str, object]) -> NLPConfig:
    nlp = _optional_mapping(raw.get("nlp"), context="nlp")
    backend_raw = nlp.get("backend", "stanza")
    if backend_raw not in {"stanza", "transformers"}:
        raise ValueError("nlp.backend must be one of: stanza, transformers")
    backend: NLPBackend = "transformers" if backend_raw == "transformers" else "stanza"

    language_raw = raw.get("language", nlp.get("language", "la"))
    package_raw = raw.get("stanza_package", nlp.get("stanza_package", nlp.get("package", "perseus")))
    cpu_only_raw = raw.get("cpu_only", nlp.get("cpu_only", True))

    if package_raw is not None and not isinstance(package_raw, (str, dict)):
        raise ValueError("stanza_package must be a string, mapping, or null")
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
        language=_str_value(language_raw, context="language"),
        stanza_package=package,
        model_name=model_name,
        cpu_only=_bool_value(cpu_only_raw, context="cpu_only", default=True),
    )


def _parse_filter_config(raw: Mapping[str, object]) -> FilterConfig:
    filters = raw.get("filter") or raw.get("filters") or {}
    filters_map = _optional_mapping(filters, context="filters")
    has_roman_singular = "roman_exceptions_file" in filters_map
    has_roman_plural = "roman_exception_files" in filters_map
    if has_roman_singular and has_roman_plural:
        raise ValueError(
            "Specify only one of filters.roman_exceptions_file and filters.roman_exception_files"
        )
    roman = (
        filters_map.get("roman_exceptions_file")
        if has_roman_singular
        else filters_map.get("roman_exception_files")
    )
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
        roman_exceptions_file=_optional_str(roman, context="filters.roman_exceptions_file"),
        upos_targets=_optional_string_set(
            raw.get("upos_targets"),
            context="upos_targets",
            default=frozenset({"NOUN"}),
        ),
        exclude_lemmas=_optional_str(
            filters_map.get("exclude_lemmas") or filters_map.get("exclude_lemmas_file"),
            context="filters.exclude_lemmas",
        ),
    )


def _parse_normalization_config(value: object) -> NormalizationConfig:
    norm = _optional_mapping(value, context="normalization")
    ligatures_raw = norm.get("ligatures")
    ligatures: Mapping[str, str] | None = None
    if ligatures_raw is not None:
        ligature_map = _as_mapping(ligatures_raw, context="normalization.ligatures")
        parsed: dict[str, str] = {}
        for key, item in ligature_map.items():
            if not isinstance(key, str) or not isinstance(item, str):
                raise ValueError("normalization.ligatures keys and values must be strings")
            parsed[key] = item
        ligatures = parsed
    return NormalizationConfig(
        enabled=_bool_value(norm.get("enabled"), context="normalization.enabled", default=True),
        casefold=_bool_value(norm.get("casefold"), context="normalization.casefold", default=False),
        uv=_optional_str(norm.get("uv"), context="normalization.uv"),
        ij=_optional_str(norm.get("ij"), context="normalization.ij"),
        diacritics=_optional_str(norm.get("diacritics"), context="normalization.diacritics"),
        ligatures=ligatures,
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
    return DictCheckConfig(
        enabled=_bool_value(dc.get("enabled"), context="dictcheck.enabled", default=False),
        wordlist=_optional_str(dc.get("wordlist"), context="dictcheck.wordlist"),
        lemma_normalize=_optional_str(dc.get("lemma_normalize"), context="dictcheck.lemma_normalize"),
    )


def _parse_ref_tags_config(value: object) -> RefTagsConfig:
    ref = _optional_mapping(value, context="ref_tags")
    patterns = ref.get("patterns") or ref.get("ref_tags_file")
    return RefTagsConfig(
        enabled=_bool_value(ref.get("enabled"), context="ref_tags.enabled", default=False),
        patterns=_optional_str(patterns, context="ref_tags.patterns"),
    )


def _parse_trace_config(value: object) -> TraceConfig:
    trace = _optional_mapping(value, context="trace")
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


def _parse_archive_config(value: object) -> ArchiveConfig:
    archive = _optional_mapping(value, context="archive")
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


def _parse_lemma_cache_config(value: object) -> LemmaCacheConfig:
    cache = _optional_mapping(value, context="lemma_cache")
    return LemmaCacheConfig(
        enabled=_bool_value(cache.get("enabled"), context="lemma_cache.enabled", default=False),
        directory=_str_value(cache.get("dir", ".lemma_cache"), context="lemma_cache.dir"),
        use_manifest=_bool_value(
            cache.get("use_manifest"),
            context="lemma_cache.use_manifest",
            default=True,
        ),
        manifest_key_mode=_str_value(
            cache.get("manifest_key_mode", "absolute"),
            context="lemma_cache.manifest_key_mode",
        ),
        lock_timeout_sec=_float_value(
            cache.get("lock_timeout_sec"),
            context="lemma_cache.lock_timeout_sec",
            default=300.0,
            minimum_exclusive=0.0,
        ),
        include_ref_tags_in_config_hash=_bool_value(
            cache.get("include_ref_tags_in_config_hash"),
            context="lemma_cache.include_ref_tags_in_config_hash",
            default=True,
        ),
    )


def _parse_prune_config(value: object) -> PruneConfig:
    prune = _optional_mapping(value, context="prune")
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


def _reject_deprecated_keys(raw: Mapping[str, object]) -> None:
    if "cleaner_config" in raw:
        raise ValueError(
            "Deprecated key 'cleaner_config' is not supported. Use preprocess: {kind: cleaner, config: ...}."
        )
    if "stanza_pkg" in raw:
        raise ValueError("Deprecated key 'stanza_pkg' is not supported. Use 'stanza_package'.")


def _build_app_config(raw: Mapping[str, object]) -> AppConfig:
    _reject_deprecated_keys(raw)
    normalized = _normalize_groups(raw)

    groups = _parse_groups(normalized["groups"])
    grouping = _parse_grouping_config(normalized.get("grouping"))
    partition_validations = _parse_partition_specs(normalized)
    if partition_validations and grouping.mode == "per_file":
        raise ValueError("validations.partitions cannot be used with grouping.mode: per_file")
    comparisons = _parse_comparison_specs(normalized)

    return AppConfig(
        groups=groups,
        preprocess=_parse_preprocess_config(normalized.get("preprocess")),
        grouping=grouping,
        nlp=_parse_nlp_config(normalized),
        filters=_parse_filter_config(normalized),
        normalization=_parse_normalization_config(normalized.get("normalization")),
        dictcheck=_parse_dictcheck_config(normalized.get("dictcheck")),
        ref_tags=_parse_ref_tags_config(normalized.get("ref_tags")),
        trace=_parse_trace_config(normalized.get("trace")),
        archive=_parse_archive_config(normalized.get("archive")),
        lemma_cache=_parse_lemma_cache_config(normalized.get("lemma_cache")),
        prune=_parse_prune_config(normalized.get("prune")),
        analysis_unit=_parse_analysis_unit(normalized.get("analysis_unit")),
        out_dir=_str_value(normalized.get("out_dir", "output"), context="out_dir"),
        csv_header=_parse_csv_header(normalized.get("csv_header")),
        vocab_path=_optional_str(normalized.get("vocab_path"), context="vocab_path"),
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


def config_to_dict(config: AppConfig) -> dict[str, object]:
    groups = {
        name: {"files": list(group.files)}
        for name, group in config.groups.items()
    }
    out: dict[str, object] = {
        "groups": groups,
        "preprocess": asdict(config.preprocess),
        "grouping": asdict(config.grouping),
        "nlp": asdict(config.nlp),
        "language": config.nlp.language,
        "stanza_package": config.nlp.stanza_package,
        "cpu_only": config.nlp.cpu_only,
        "filters": {
            "min_token_length": config.filters.min_token_length,
            "drop_roman_numerals": config.filters.drop_roman_numerals,
            "roman_exceptions_file": config.filters.roman_exceptions_file,
            "exclude_lemmas": config.filters.exclude_lemmas,
        },
        "upos_targets": sorted(config.filters.upos_targets),
        "normalization": asdict(config.normalization),
        "dictcheck": asdict(config.dictcheck),
        "ref_tags": asdict(config.ref_tags),
        "trace": {
            "enabled": config.trace.enabled,
            "path": config.trace.path,
            "max_rows": config.trace.max_rows,
            "only_keys": sorted(config.trace.only_keys),
            "write_truncation_marker": config.trace.write_truncation_marker,
        },
        "archive": asdict(config.archive),
        "lemma_cache": {
            "enabled": config.lemma_cache.enabled,
            "dir": config.lemma_cache.directory,
            "use_manifest": config.lemma_cache.use_manifest,
            "manifest_key_mode": config.lemma_cache.manifest_key_mode,
            "lock_timeout_sec": config.lemma_cache.lock_timeout_sec,
            "include_ref_tags_in_config_hash": config.lemma_cache.include_ref_tags_in_config_hash,
        },
        "prune": asdict(config.prune),
        "analysis_unit": config.analysis_unit,
        "out_dir": config.out_dir,
        "vocab_path": config.vocab_path,
        "comparisons": [asdict(spec) for spec in config.comparisons],
        "validations": {
            "partitions": [asdict(spec) for spec in config.partition_validations],
        },
    }
    if config.csv_header is not None:
        out["csv_header"] = list(config.csv_header)
    return out


def unknown_top_level_keys(raw_config: Mapping[str, object]) -> list[str]:
    return [str(key) for key in raw_config if key not in KNOWN_TOP_LEVEL_KEYS]


def unknown_filter_keys(raw_config: Mapping[str, object]) -> list[str]:
    filters = raw_config.get("filter") or raw_config.get("filters") or {}
    if not isinstance(filters, Mapping):
        return []
    return [str(key) for key in filters if key not in KNOWN_FILTER_KEYS]
