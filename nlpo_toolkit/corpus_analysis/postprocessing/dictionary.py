from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import AbstractSet


@dataclass(frozen=True)
class DictionaryClassification:
    known: Counter[str]
    unknown: Counter[str]


def classify_dictionary_entries(
    counter: Counter[str], known_words: AbstractSet[str]
) -> DictionaryClassification:
    return DictionaryClassification(
        known=Counter({word: count for word, count in counter.items() if word in known_words}),
        unknown=Counter({word: count for word, count in counter.items() if word not in known_words}),
    )
