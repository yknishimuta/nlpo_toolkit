from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

from nlpo_toolkit.comparison.configured import ComparisonSpec
from ..partition_models import PartitionSpec


AnalysisUnit = Literal["lemma", "surface"]
GroupingMode = Literal["groups", "per_file", "auto_single_cleaned"]
NLPBackendName = Literal["stanza", "transformers"]


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
    backend: NLPBackendName = "stanza"
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
    analysis_unit: AnalysisUnit = "lemma"
    out_dir: str = "output"
    csv_header: tuple[str, str] | None = None
    comparisons: tuple[ComparisonSpec, ...] = ()
    partition_validations: tuple[PartitionSpec, ...] = ()
