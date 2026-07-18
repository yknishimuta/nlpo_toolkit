from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal

from ..analysis_records import NLPAnalysisRecord
from ..corpus import PreparedCorpus
from ..requests import CorpusPreparationRequest
from .errors import FeatureError
from nlpo_toolkit.immutable_collections import freeze_mapping


FeatureField = Literal["lemma", "token"]
FeatureScalar = str | int | float


@dataclass(frozen=True)
class MorphologyOptions:
    enabled: bool = False
    attributes: tuple[str, ...] = ()
    bundle_top: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise FeatureError("morphology enabled must be a bool")
        attributes = tuple(self.attributes)
        if any(not isinstance(item, str) or not item.strip() for item in attributes):
            raise FeatureError("--morph-attribute must not be empty")
        attributes = tuple(item.strip() for item in attributes)
        if len(attributes) != len(set(attributes)):
            raise FeatureError("duplicate morphology attribute")
        object.__setattr__(self, "attributes", attributes)
        if self.bundle_top is not None and (
            isinstance(self.bundle_top, bool)
            or not isinstance(self.bundle_top, int)
            or self.bundle_top <= 0
        ):
            raise FeatureError("--morph-bundle-top must be a positive integer")


@dataclass(frozen=True)
class CharacterNgramOptions:
    sizes: tuple[int, ...]
    top: int = 500

    def __post_init__(self) -> None:
        object.__setattr__(self, "sizes", tuple(self.sizes))
        if not self.sizes:
            raise FeatureError("--char-ngram-size must be specified")
        for size in self.sizes:
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise FeatureError("--char-ngram-size must be a positive integer")
        if len(self.sizes) != len(set(self.sizes)):
            duplicate = next(size for size in self.sizes if self.sizes.count(size) > 1)
            raise FeatureError(f"duplicate character n-gram size: {duplicate}")
        if isinstance(self.top, bool) or not isinstance(self.top, int) or self.top <= 0:
            raise FeatureError("--char-ngram-top must be a positive integer")


@dataclass(frozen=True)
class UposNgramOptions:
    sizes: tuple[int, ...]
    top: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "sizes", tuple(self.sizes))
        if not self.sizes:
            raise FeatureError("--upos-ngram-size must be specified")
        for size in self.sizes:
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or size not in (2, 3)
            ):
                raise FeatureError("--upos-ngram-size must be 2 or 3")
        if len(self.sizes) != len(set(self.sizes)):
            duplicate = next(size for size in self.sizes if self.sizes.count(size) > 1)
            raise FeatureError(f"duplicate UPOS n-gram size: {duplicate}")
        if isinstance(self.top, bool) or not isinstance(self.top, int) or self.top <= 0:
            raise FeatureError("--upos-ngram-top must be a positive integer")


@dataclass(frozen=True)
class LexicalDiversityOptions:
    window_size: int = 100
    mtld_threshold: float = 0.72
    hdd_sample_size: int = 42

    def __post_init__(self) -> None:
        for name, value in (
            ("window_size", self.window_size),
            ("hdd_sample_size", self.hdd_sample_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise FeatureError(f"{name} must be a positive integer")
        threshold = self.mtld_threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0.0 < threshold < 1.0
        ):
            raise FeatureError("mtld_threshold must be a finite number between 0 and 1")
        object.__setattr__(self, "mtld_threshold", float(threshold))


@dataclass(frozen=True)
class FunctionWordSource:
    path: Path
    field: FeatureField = "lemma"

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise FeatureError("function-word path must be a Path")
        if self.field not in {"lemma", "token"}:
            raise FeatureError("--function-word-field must be 'lemma' or 'token'")


@dataclass(frozen=True)
class FunctionWordVocabulary:
    terms: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        if not self.terms:
            raise FeatureError("function-word list contains no terms")


@dataclass(frozen=True)
class FunctionWordOptions:
    vocabulary: FunctionWordVocabulary
    field: FeatureField = "lemma"

    def __post_init__(self) -> None:
        if self.field not in {"lemma", "token"}:
            raise FeatureError("--function-word-field must be 'lemma' or 'token'")


@dataclass(frozen=True)
class FeatureSamplingOptions:
    window_tokens: int | None = None
    step_tokens: int | None = None
    include_partial: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("window_tokens", self.window_tokens),
            ("step_tokens", self.step_tokens),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise FeatureError(f"{name} must be a positive integer")
        if self.window_tokens is None and self.step_tokens is not None:
            raise FeatureError("step_tokens requires window_tokens")
        if not isinstance(self.include_partial, bool):
            raise FeatureError("include_partial must be a bool")

    @property
    def enabled(self) -> bool:
        return self.window_tokens is not None

    @property
    def effective_step_tokens(self) -> int | None:
        if self.window_tokens is None:
            return None
        return self.step_tokens or self.window_tokens


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
        object.__setattr__(
            self,
            "roman_exceptions",
            frozenset(
                str(item).strip().lower()
                for item in self.roman_exceptions
                if str(item).strip()
            ),
        )


@dataclass(frozen=True)
class FeatureOptions:
    field: FeatureField = "lemma"
    mfw: int = 0
    include_upos: bool = True
    include_basic: bool = True
    filter_policy: FeatureFilterPolicy = FeatureFilterPolicy()
    sampling: FeatureSamplingOptions = FeatureSamplingOptions()
    lexical_diversity: LexicalDiversityOptions | None = None
    function_words: FunctionWordOptions | None = None
    character_ngrams: CharacterNgramOptions | None = None
    upos_ngrams: UposNgramOptions | None = None
    morphology: MorphologyOptions | None = None


def validate_feature_options(options: FeatureOptions) -> None:
    if options.mfw < 0:
        raise FeatureError("--mfw must be non-negative")
    if options.field not in {"lemma", "token"}:
        raise FeatureError("--field must be 'lemma' or 'token'")


@dataclass(frozen=True)
class FeatureRequest:
    corpus: CorpusPreparationRequest
    field: FeatureField = "lemma"
    mfw: int = 0
    include_upos: bool = True
    include_basic: bool = True
    sampling: FeatureSamplingOptions = FeatureSamplingOptions()
    lexical_diversity: LexicalDiversityOptions | None = None
    function_words: FunctionWordSource | None = None
    character_ngrams: CharacterNgramOptions | None = None
    upos_ngrams: UposNgramOptions | None = None
    morphology: MorphologyOptions | None = None


@dataclass(frozen=True)
class FeatureSampleMetadata:
    source_file: str
    sample_id: str
    sample_index: int
    start_token: int
    end_token: int
    kind: Literal["full", "partial"]


@dataclass(frozen=True)
class AnalyzedFeatureCorpus:
    source: PreparedCorpus
    raw_records: tuple[NLPAnalysisRecord, ...]
    lexical_records: tuple[NLPAnalysisRecord, ...]
    char_count: int | None = None
    sample: FeatureSampleMetadata | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_records", tuple(self.raw_records))
        object.__setattr__(self, "lexical_records", tuple(self.lexical_records))

    @property
    def raw_record_count(self) -> int:
        return len(self.raw_records)

    @property
    def lexical_record_count(self) -> int:
        return len(self.lexical_records)

    @property
    def sentence_count(self) -> int:
        return len(
            {(record.chunk_index, record.sentence_index) for record in self.raw_records}
        )


@dataclass(frozen=True)
class FeatureRow(Mapping[str, FeatureScalar]):
    values: Mapping[str, FeatureScalar]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", freeze_mapping(self.values))

    @classmethod
    def from_mapping(cls, values: Mapping[str, FeatureScalar]) -> FeatureRow:
        return cls(values)

    def __getitem__(self, key: str) -> FeatureScalar:
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class FeatureCommandResult:
    rows: tuple[FeatureRow, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))
