from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_serializer,
    field_validator,
    model_validator,
)

from nlpo_toolkit.comparison.configured import ComparisonSpec
from nlpo_toolkit.config_model import (
    ConfigModel,
    NonBlankStr,
    PositiveFiniteFloat,
)
from ..partition_models import PartitionSpec


AnalysisUnit = Literal["lemma", "surface"]
GroupingMode = Literal["groups", "per_file", "auto_single_cleaned"]
NLPBackendName = Literal["stanza", "transformers"]
NonNegativeStrictInt = Annotated[StrictInt, Field(ge=0)]


def uppercase_non_blank(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("must be a non-empty string")
    return stripped.upper()


UpperNonBlankStr = Annotated[StrictStr, AfterValidator(uppercase_non_blank)]


class GroupConfig(ConfigModel):
    files: tuple[NonBlankStr, ...]


class PreprocessConfig(ConfigModel):
    kind: Literal["cleaner"] | None = None
    config: NonBlankStr | None = None

    @model_validator(mode="after")
    def validate_pair(self) -> PreprocessConfig:
        if (self.kind is None) != (self.config is None):
            raise ValueError("kind and config must be specified together")
        return self


class GroupingConfig(ConfigModel):
    mode: GroupingMode = "groups"
    auto_group_name: NonBlankStr = "text"


class NLPConfig(ConfigModel):
    backend: NLPBackendName = "stanza"
    language: NonBlankStr = "la"
    stanza_package: NonBlankStr | dict[StrictStr, StrictStr] | None = "perseus"
    model_name: NonBlankStr | None = None
    cpu_only: StrictBool = True

    @field_validator("stanza_package", mode="before")
    @classmethod
    def default_null_package(cls, value: object) -> object:
        return "perseus" if value is None else value

    @model_validator(mode="after")
    def validate_backend(self) -> NLPConfig:
        if self.backend == "transformers" and self.model_name is None:
            raise ValueError("model_name is required when backend=transformers")
        return self


class FilterConfig(ConfigModel):
    min_token_length: NonNegativeStrictInt = 0
    drop_roman_numerals: StrictBool = False
    roman_exceptions_file: NonBlankStr | None = None
    upos_targets: frozenset[UpperNonBlankStr] = frozenset({"NOUN"})

    @field_serializer("upos_targets")
    def serialize_upos_targets(self, value: frozenset[str]) -> list[str]:
        return sorted(value)


class NormalizationConfig(ConfigModel):
    enabled: StrictBool = True
    casefold: StrictBool = False
    map_u_v: StrictBool = False
    map_i_j: StrictBool = False
    strip_diacritics: StrictBool = False
    normalize_ligatures: StrictBool = False
    unicode_nf: NonBlankStr | None = None


class DictCheckConfig(ConfigModel):
    enabled: StrictBool = False
    wordlist: NonBlankStr | None = None
    lemma_normalize: NonBlankStr | None = None


class RefTagsConfig(ConfigModel):
    enabled: StrictBool = False
    patterns: NonBlankStr | None = None

    @model_validator(mode="after")
    def validate_patterns(self) -> RefTagsConfig:
        if self.enabled and self.patterns is None:
            raise ValueError("patterns is required when enabled=true")
        return self


class TraceConfig(ConfigModel):
    enabled: StrictBool = False
    path: NonBlankStr | None = None
    max_rows: NonNegativeStrictInt | None = 0
    only_keys: frozenset[UpperNonBlankStr] = frozenset()
    write_truncation_marker: StrictBool = True

    @field_serializer("only_keys")
    def serialize_only_keys(self, value: frozenset[str]) -> list[str]:
        return sorted(value)


class TokenArtifactConfig(ConfigModel):
    enabled: StrictBool = False
    path: NonBlankStr = "output/tokens.tsv"


class ArtifactsConfig(ConfigModel):
    tokens: TokenArtifactConfig = Field(default_factory=TokenArtifactConfig)


class ArchiveConfig(ConfigModel):
    enabled: StrictBool = False
    runs_dir: NonBlankStr = "runs"
    include_input: StrictBool = False
    include_cleaned: StrictBool = False


class AnalysisCacheConfig(ConfigModel):
    enabled: StrictBool = False
    directory: NonBlankStr = Field(
        default=".analysis_cache",
        validation_alias="dir",
        serialization_alias="dir",
    )
    lock_timeout_sec: PositiveFiniteFloat = 300.0


class ValidationsConfig(ConfigModel):
    partitions: tuple[PartitionSpec, ...] = ()


class AppConfig(ConfigModel):
    groups: dict[NonBlankStr, GroupConfig]
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    grouping: GroupingConfig = Field(default_factory=GroupingConfig)
    nlp: NLPConfig = Field(default_factory=NLPConfig)
    filters: FilterConfig = Field(default_factory=FilterConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    dictcheck: DictCheckConfig = Field(default_factory=DictCheckConfig)
    ref_tags: RefTagsConfig = Field(default_factory=RefTagsConfig)
    trace: TraceConfig = Field(default_factory=TraceConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    archive: ArchiveConfig = Field(default_factory=ArchiveConfig)
    analysis_cache: AnalysisCacheConfig = Field(default_factory=AnalysisCacheConfig)
    analysis_unit: AnalysisUnit = "lemma"
    out_dir: NonBlankStr = "output"
    csv_header: tuple[NonBlankStr, NonBlankStr] | None = None
    comparisons: tuple[ComparisonSpec, ...] = ()
    validations: ValidationsConfig = Field(default_factory=ValidationsConfig)

    @model_validator(mode="after")
    def validate_spec_structure(self) -> AppConfig:
        comparison_names: set[str] = set()
        for spec in self.comparisons:
            if spec.name in comparison_names:
                raise ValueError(f"duplicate comparison name: {spec.name}")
            comparison_names.add(spec.name)

        partition_names: set[str] = set()
        for spec in self.validations.partitions:
            if spec.name in partition_names:
                raise ValueError(f"duplicate partition name: {spec.name}")
            partition_names.add(spec.name)

        if self.grouping.mode == "per_file" and self.comparisons:
            raise ValueError("comparisons cannot be used with grouping.mode=per_file")
        if self.grouping.mode == "per_file" and self.validations.partitions:
            raise ValueError(
                "validations.partitions cannot be used with grouping.mode=per_file"
            )
        return self

    def to_external_dict(self) -> dict[str, object]:
        data = self.model_dump(mode="json", by_alias=True)
        if self.preprocess.kind is None:
            data.pop("preprocess", None)
        if self.csv_header is None:
            data.pop("csv_header", None)
        return data
