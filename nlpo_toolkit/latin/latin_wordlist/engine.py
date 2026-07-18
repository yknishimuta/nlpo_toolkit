from __future__ import annotations

import string
from collections.abc import Iterator, Mapping

from .models import (
    ConlluCandidates,
    ExtraWordlistCandidates,
    TextCandidates,
    WordlistFilterPolicy,
    WordlistTokenizationPolicy,
)


def iter_latin_word_candidates(
    text: str, *, policy: WordlistTokenizationPolicy, min_length: int
) -> Iterator[str]:
    punctuation = str.maketrans(
        {character: " " for character in string.punctuation + policy.extra_punct}
    )
    for candidate in text.translate(punctuation).split():
        word = candidate.lower()
        if len(word) >= min_length and word.isalpha():
            yield word


def select_frequent_forms(
    counts: Mapping[str, int], *, minimum_frequency: int
) -> frozenset[str]:
    return frozenset(
        word for word, count in counts.items() if count >= minimum_frequency
    )


def merge_wordlist_candidates(
    *,
    conllu: ConlluCandidates,
    text: TextCandidates,
    extras: tuple[ExtraWordlistCandidates, ...],
    filters: WordlistFilterPolicy,
) -> tuple[str, ...]:
    entries = set(conllu.lemmas)
    entries.update(
        select_frequent_forms(
            conllu.form_counts, minimum_frequency=filters.min_form_freq
        )
    )
    entries.update(
        select_frequent_forms(text.form_counts, minimum_frequency=filters.min_text_freq)
    )
    for extra in extras:
        entries.update(extra.entries)
    return tuple(sorted(entries))
