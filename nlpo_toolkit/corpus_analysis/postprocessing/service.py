from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from nlpo_toolkit.immutable_collections import freeze_count_mapping, freeze_mapping

import nlpo_toolkit.nlp.vocabulary as vocabulary

from .. import dictcheck
from ..planning.models import AnalysisPlan
from .dictionary import DictionaryClassification, classify_dictionary_entries
from .lemma_normalization import apply_lemma_normalization


@dataclass(frozen=True)
class GroupPostprocessingResult:
    counter: Mapping[str, int]
    dictionary: DictionaryClassification | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "counter", freeze_count_mapping(self.counter))


@dataclass(frozen=True)
class PostprocessingResources:
    normalization_map: Mapping[str, str] | None
    known_words: frozenset[str] | None

    def __post_init__(self) -> None:
        if self.normalization_map is not None:
            object.__setattr__(self, "normalization_map", freeze_mapping(self.normalization_map))
        if self.known_words is not None:
            object.__setattr__(self, "known_words", frozenset(self.known_words))


def load_postprocessing_resources(plan: AnalysisPlan) -> PostprocessingResources:
    normalization_path = plan.config_files.path("dictcheck.lemma_normalize")
    normalization_map = (
        dictcheck.load_lemma_normalize_map(normalization_path)
        if normalization_path is not None
        else None
    )
    wordlist_path = plan.config_files.path("dictcheck.wordlist")
    if plan.config.dictcheck.enabled and wordlist_path is None:
        raise ValueError(
            "dictcheck.wordlist is required when dictcheck.enabled=true "
            f"(analysis_unit={plan.analysis_mode.unit})"
        )
    known_words = (
        vocabulary.load_wordlist(wordlist_path)
        if plan.config.dictcheck.enabled
        else None
    )
    return PostprocessingResources(normalization_map, known_words)


def postprocess_group_counter(
    counter: Mapping[str, int],
    *,
    resources: PostprocessingResources,
) -> GroupPostprocessingResult:
    normalized = apply_lemma_normalization(counter, resources.normalization_map)
    dictionary = (
        classify_dictionary_entries(normalized, resources.known_words)
        if resources.known_words is not None
        else None
    )
    return GroupPostprocessingResult(normalized, dictionary)
