from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping

import nlpo_toolkit.nlp.vocabulary as vocabulary

from .. import dictcheck
from ..planning.models import AnalysisPlan
from .dictionary import DictionaryClassification, classify_dictionary_entries
from .lemma_normalization import apply_lemma_normalization


@dataclass(frozen=True)
class GroupPostprocessingResult:
    counter: Counter[str]
    dictionary: DictionaryClassification | None


@dataclass(frozen=True)
class PostprocessingResources:
    normalization_map: Mapping[str, str] | None
    known_words: frozenset[str] | None


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
    counter: Counter[str],
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
