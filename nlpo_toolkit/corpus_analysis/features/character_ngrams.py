from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from nlpo_toolkit.immutable_collections import freeze_mapping

from .character_text import encode_character_ngram, normalize_character_stream
from .errors import FeatureError
from .models import CharacterNgramMode, CharacterNgramOptions, FeatureScalar


def character_ngram_column_prefix(*, mode: CharacterNgramMode, size: int) -> str:
    prefixes = {
        CharacterNgramMode.FULL: f"char{size}_",
        CharacterNgramMode.NO_PUNCTUATION: f"char_nopunct{size}_",
        CharacterNgramMode.LETTERS_SPACES: f"char_letters_spaces{size}_",
        CharacterNgramMode.LETTERS_ONLY: f"char_letters_only{size}_",
    }
    return prefixes[mode]


def character_ngram_column_name(
    *, mode: CharacterNgramMode, size: int, value: str
) -> str:
    return character_ngram_column_prefix(mode=mode, size=size) + encode_character_ngram(value)


@dataclass(frozen=True)
class CharacterNgramTerm:
    size: int
    value: str
    column_name: str
    mode: CharacterNgramMode = CharacterNgramMode.FULL

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
        if not isinstance(self.mode, CharacterNgramMode):
            raise FeatureError("character n-gram term mode is invalid")
        expected = character_ngram_column_name(
            mode=self.mode, size=self.size, value=self.value
        )
        if self.column_name != expected or not self.column_name.isascii():
            raise FeatureError("character n-gram column name is not canonical")


@dataclass(frozen=True)
class CharacterNgramVocabulary:
    terms: tuple[CharacterNgramTerm, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        if not self.terms:
            raise FeatureError("character n-gram vocabulary must not be empty")
        keys = tuple((term.mode, term.size, term.value) for term in self.terms)
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
    terms = []
    for mode in options.modes:
        streams = tuple(normalize_character_stream(text, mode=mode) for text in texts)
        for size in options.sizes:
            frequencies: Counter[str] = Counter()
            for stream in streams:
                frequencies.update(iter_character_ngrams(stream, size=size))
            if not frequencies:
                raise FeatureError(
                    f"no character {size}-grams can be generated for mode "
                    f"{mode.value!r} from the prepared corpora"
                )
            selected = sorted(
                frequencies, key=lambda value: (-frequencies[value], value)
            )[: options.top]
            terms.extend(
                CharacterNgramTerm(
                    size=size,
                    value=value,
                    column_name=character_ngram_column_name(
                        mode=mode, size=size, value=value
                    ),
                    mode=mode,
                )
                for value in selected
            )
    return CharacterNgramVocabulary(tuple(terms))


def compute_character_ngram_features(
    text: str,
    *,
    vocabulary: CharacterNgramVocabulary,
) -> Mapping[str, FeatureScalar]:
    modes = tuple(dict.fromkeys(term.mode for term in vocabulary.terms))
    streams = {mode: normalize_character_stream(text, mode=mode) for mode in modes}
    keys = tuple(dict.fromkeys((term.mode, term.size) for term in vocabulary.terms))
    counts = {
        key: Counter(iter_character_ngrams(streams[key[0]], size=key[1]))
        for key in keys
    }
    denominators = {
        key: max(len(streams[key[0]]) - key[1] + 1, 0) for key in keys
    }
    return freeze_mapping(
        {
            term.column_name: float(
                counts[(term.mode, term.size)].get(term.value, 0)
                / denominators[(term.mode, term.size)]
                if denominators[(term.mode, term.size)]
                else 0.0
            )
            for term in vocabulary.terms
        }
    )
