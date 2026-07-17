from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Set
from dataclasses import dataclass
from nlpo_toolkit.immutable_collections import freeze_count_mapping


@dataclass(frozen=True)
class DictionaryClassification:
    known: Mapping[str, int]
    unknown: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "known", freeze_count_mapping(self.known))
        object.__setattr__(self, "unknown", freeze_count_mapping(self.unknown))


def classify_dictionary_entries(
    counter: Mapping[str, int], known_words: Set[str]
) -> DictionaryClassification:
    return DictionaryClassification(
        known=Counter({word: count for word, count in counter.items() if word in known_words}),
        unknown=Counter({word: count for word, count in counter.items() if word not in known_words}),
    )
