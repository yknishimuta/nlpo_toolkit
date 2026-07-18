from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from nlpo_toolkit.immutable_collections import freeze_mapping

from ..analysis_records import NLPAnalysisRecord
from .errors import FeatureError
from .models import (
    AnalyzedFeatureCorpus,
    FeatureScalar,
    UposNgramOptions,
)


def normalize_upos_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    return normalized or None


def iter_upos_runs(
    records: Sequence[NLPAnalysisRecord],
) -> Iterator[tuple[str, ...]]:
    current_sentence: tuple[int, int] | None = None
    completed_sentences: set[tuple[int, int]] = set()
    run: list[str] = []
    for record in records:
        sentence = (record.chunk_index, record.sentence_index)
        if current_sentence is None:
            if sentence in completed_sentences:
                raise FeatureError(
                    f"UPOS sentence ID reappeared out of order: {sentence}"
                )
            current_sentence = sentence
        elif sentence != current_sentence:
            if run:
                yield tuple(run)
            run = []
            completed_sentences.add(current_sentence)
            if sentence in completed_sentences:
                raise FeatureError(
                    f"UPOS sentence ID reappeared out of order: {sentence}"
                )
            current_sentence = sentence
        tag = normalize_upos_value(record.upos)
        if tag is None:
            if run:
                yield tuple(run)
                run = []
            continue
        run.append(tag)
    if run:
        yield tuple(run)


def iter_upos_ngrams(run: Sequence[str], *, size: int) -> Iterator[tuple[str, ...]]:
    if isinstance(size, bool) or not isinstance(size, int) or size not in (2, 3):
        raise FeatureError("UPOS n-gram size must be 2 or 3")
    for index in range(max(len(run) - size + 1, 0)):
        yield tuple(run[index : index + size])


def _encode_tag(tag: str) -> str:
    parts = []
    for character in tag:
        if "A" <= character <= "Z" or "0" <= character <= "9":
            parts.append(character)
        else:
            parts.append(f"_u{ord(character):06x}_")
    return "".join(parts)


def _column_name(size: int, tags: tuple[str, ...]) -> str:
    return f"upos{size}_{'_'.join(_encode_tag(tag) for tag in tags)}"


@dataclass(frozen=True)
class UposNgramTerm:
    size: int
    tags: tuple[str, ...]
    column_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", tuple(self.tags))
        if self.size not in (2, 3) or len(self.tags) != self.size:
            raise FeatureError("UPOS n-gram term size does not match its tags")
        canonical = tuple(normalize_upos_value(tag) for tag in self.tags)
        if any(tag is None for tag in canonical) or canonical != self.tags:
            raise FeatureError("UPOS n-gram tags must be canonical and non-empty")
        if self.column_name != _column_name(self.size, self.tags):
            raise FeatureError("UPOS n-gram column name is not canonical")
        if not self.column_name.isascii():
            raise FeatureError("UPOS n-gram column name must be ASCII")


@dataclass(frozen=True)
class UposNgramVocabulary:
    terms: tuple[UposNgramTerm, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        if not self.terms:
            raise FeatureError("UPOS n-gram vocabulary must not be empty")
        keys = tuple((term.size, term.tags) for term in self.terms)
        columns = tuple(term.column_name for term in self.terms)
        if len(keys) != len(set(keys)):
            raise FeatureError("duplicate UPOS n-gram vocabulary term")
        if len(columns) != len(set(columns)):
            duplicate = next(name for name in columns if columns.count(name) > 1)
            raise FeatureError(f"duplicate UPOS n-gram feature column: {duplicate}")


def _counts_for_size(
    records: Sequence[NLPAnalysisRecord], *, size: int
) -> Counter[tuple[str, ...]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for run in iter_upos_runs(records):
        counts.update(iter_upos_ngrams(run, size=size))
    return counts


def select_upos_ngram_vocabulary(
    corpora: Sequence[AnalyzedFeatureCorpus],
    *,
    options: UposNgramOptions,
) -> UposNgramVocabulary:
    terms = []
    for size in options.sizes:
        frequencies: Counter[tuple[str, ...]] = Counter()
        for corpus in corpora:
            frequencies.update(_counts_for_size(corpus.lexical_records, size=size))
        if not frequencies:
            raise FeatureError(
                f"no UPOS {size}-grams can be generated from the analyzed corpora"
            )
        selected = sorted(frequencies, key=lambda tags: (-frequencies[tags], tags))[
            : options.top
        ]
        terms.extend(
            UposNgramTerm(size, tags, _column_name(size, tags)) for tags in selected
        )
    return UposNgramVocabulary(tuple(terms))


def compute_upos_ngram_features(
    records: Sequence[NLPAnalysisRecord],
    *,
    vocabulary: UposNgramVocabulary,
) -> Mapping[str, FeatureScalar]:
    sizes = tuple(dict.fromkeys(term.size for term in vocabulary.terms))
    counts = {size: _counts_for_size(records, size=size) for size in sizes}
    denominators = {size: sum(counts[size].values()) for size in sizes}
    return freeze_mapping(
        {
            term.column_name: float(
                counts[term.size].get(term.tags, 0) / denominators[term.size]
                if denominators[term.size]
                else 0.0
            )
            for term in vocabulary.terms
        }
    )
