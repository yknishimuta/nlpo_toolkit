from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from nlpo_toolkit.immutable_collections import freeze_mapping

from .character_ngrams import iter_character_ngrams
from .character_text import encode_character_ngram
from .content_masking import build_content_masked_character_stream
from .errors import FeatureError
from .models import (
    AnalyzedFeatureCorpus,
    ContentMaskedCharacterNgramOptions,
    FeatureScalar,
)


def content_masked_character_ngram_column_name(*, size: int, value: str) -> str:
    return f"char_masked{size}_" + encode_character_ngram(value)


@dataclass(frozen=True)
class ContentMaskedCharacterNgramTerm:
    size: int
    value: str
    column_name: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size <= 0
        ):
            raise FeatureError("content-masked character n-gram size must be positive")
        if not self.value or len(self.value) != self.size:
            raise FeatureError(
                "content-masked character n-gram value must match its size"
            )
        expected = content_masked_character_ngram_column_name(
            size=self.size, value=self.value
        )
        if self.column_name != expected or not self.column_name.isascii():
            raise FeatureError(
                "content-masked character n-gram column name is not canonical"
            )


@dataclass(frozen=True)
class ContentMaskedCharacterNgramVocabulary:
    terms: tuple[ContentMaskedCharacterNgramTerm, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        if not self.terms:
            raise FeatureError(
                "content-masked character n-gram vocabulary must not be empty"
            )
        keys = tuple((term.size, term.value) for term in self.terms)
        columns = tuple(term.column_name for term in self.terms)
        if len(keys) != len(set(keys)):
            raise FeatureError(
                "duplicate content-masked character n-gram vocabulary term"
            )
        if len(columns) != len(set(columns)):
            duplicate = next(name for name in columns if columns.count(name) > 1)
            raise FeatureError(
                f"duplicate content-masked character n-gram feature column: {duplicate}"
            )


def select_content_masked_character_ngram_vocabulary(
    corpora: Sequence[AnalyzedFeatureCorpus],
    *,
    options: ContentMaskedCharacterNgramOptions,
) -> ContentMaskedCharacterNgramVocabulary:
    streams = tuple(build_content_masked_character_stream(corpus) for corpus in corpora)
    terms = []
    for size in options.sizes:
        frequencies: Counter[str] = Counter()
        for stream in streams:
            frequencies.update(iter_character_ngrams(stream, size=size))
        if not frequencies:
            raise FeatureError(
                f"no content-masked character {size}-grams can be generated "
                "from the analyzed corpora"
            )
        selected = sorted(frequencies, key=lambda value: (-frequencies[value], value))[
            : options.top
        ]
        terms.extend(
            ContentMaskedCharacterNgramTerm(
                size,
                value,
                content_masked_character_ngram_column_name(size=size, value=value),
            )
            for value in selected
        )
    return ContentMaskedCharacterNgramVocabulary(tuple(terms))


def compute_content_masked_character_ngram_features(
    corpus: AnalyzedFeatureCorpus,
    *,
    vocabulary: ContentMaskedCharacterNgramVocabulary,
) -> Mapping[str, FeatureScalar]:
    stream = build_content_masked_character_stream(corpus)
    sizes = tuple(dict.fromkeys(term.size for term in vocabulary.terms))
    counts = {size: Counter(iter_character_ngrams(stream, size=size)) for size in sizes}
    denominators = {size: max(len(stream) - size + 1, 0) for size in sizes}
    return freeze_mapping(
        {
            term.column_name: float(
                counts[term.size].get(term.value, 0) / denominators[term.size]
                if denominators[term.size]
                else 0.0
            )
            for term in vocabulary.terms
        }
    )
