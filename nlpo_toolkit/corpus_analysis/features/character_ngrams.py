from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from nlpo_toolkit.immutable_collections import freeze_mapping

from .character_text import encode_character_ngram, normalize_character_stream
from .errors import FeatureError
from .models import CharacterNgramOptions, FeatureScalar


@dataclass(frozen=True)
class CharacterNgramTerm:
    size: int
    value: str
    column_name: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size <= 0
        ):
            raise FeatureError("character n-gram size must be positive")
        if not self.value or len(self.value) != self.size:
            raise FeatureError("character n-gram value must match its size")
        if not self.column_name:
            raise FeatureError("character n-gram column name must not be empty")
        expected = f"char{self.size}_{encode_character_ngram(self.value)}"
        if self.column_name != expected or not self.column_name.isascii():
            raise FeatureError("character n-gram column name is not canonical")


@dataclass(frozen=True)
class CharacterNgramVocabulary:
    terms: tuple[CharacterNgramTerm, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        if not self.terms:
            raise FeatureError("character n-gram vocabulary must not be empty")
        keys = tuple((term.size, term.value) for term in self.terms)
        columns = tuple(term.column_name for term in self.terms)
        if len(keys) != len(set(keys)):
            raise FeatureError("duplicate character n-gram vocabulary term")
        if len(columns) != len(set(columns)):
            duplicate = next(name for name in columns if columns.count(name) > 1)
            raise FeatureError(
                f"duplicate character n-gram feature column: {duplicate}"
            )


def iter_character_ngrams(text: str, *, size: int) -> Iterator[str]:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise FeatureError("character n-gram size must be positive")
    for index in range(max(len(text) - size + 1, 0)):
        yield text[index : index + size]


def select_character_ngram_vocabulary(
    texts: Sequence[str],
    *,
    options: CharacterNgramOptions,
) -> CharacterNgramVocabulary:
    streams = tuple(normalize_character_stream(text) for text in texts)
    terms = []
    for size in options.sizes:
        frequencies: Counter[str] = Counter()
        for stream in streams:
            frequencies.update(iter_character_ngrams(stream, size=size))
        if not frequencies:
            raise FeatureError(
                f"no character {size}-grams can be generated from the prepared corpora"
            )
        selected = sorted(frequencies, key=lambda value: (-frequencies[value], value))[
            : options.top
        ]
        terms.extend(
            CharacterNgramTerm(
                size=size,
                value=value,
                column_name=f"char{size}_{encode_character_ngram(value)}",
            )
            for value in selected
        )
    return CharacterNgramVocabulary(tuple(terms))


def compute_character_ngram_features(
    text: str,
    *,
    vocabulary: CharacterNgramVocabulary,
) -> Mapping[str, FeatureScalar]:
    stream = normalize_character_stream(text)
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
