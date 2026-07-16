from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

import nlpo_toolkit.nlp.vocabulary as vocabulary

from .corpus import PreparedCorpus
from . import dictcheck
from .outputs import write_frequency_csv
from .planning.models import AnalysisPlan
from .artifacts.models import ArtifactKind, ArtifactPlan

__all__ = [
    "KnownUnknownCounters",
    "apply_lemma_normalization",
    "load_configured_known_words",
    "load_configured_lemma_normalization",
    "split_known_unknown",
    "write_dictcheck_outputs",
    "write_group_analysis_outputs",
]


@dataclass(frozen=True)
class KnownUnknownCounters:
    known: Counter[str]
    unknown: Counter[str]


def apply_lemma_normalization(
    counter: Mapping[str, int], normalization_map: Mapping[str, str]
) -> Counter[str]:
    normalized: Counter[str] = Counter()
    for lemma, count in counter.items():
        normalized[normalization_map.get(lemma, lemma)] += count
    return normalized


def split_known_unknown(
    counter: Mapping[str, int], known_words: Iterable[str]
) -> KnownUnknownCounters:
    known_set = set(known_words)
    return KnownUnknownCounters(
        known=Counter({word: count for word, count in counter.items() if word in known_set}),
        unknown=Counter({word: count for word, count in counter.items() if word not in known_set}),
    )


def load_configured_lemma_normalization(plan: AnalysisPlan) -> Mapping[str, str] | None:
    path = plan.config_files.path("dictcheck.lemma_normalize")
    return dictcheck.load_lemma_normalize_map(path) if path is not None else None


def load_configured_known_words(plan: AnalysisPlan) -> frozenset[str] | None:
    path = plan.config_files.path("dictcheck.wordlist")
    if plan.config.dictcheck.enabled and path is None:
        raise ValueError(
            "dictcheck.wordlist is required when dictcheck.enabled=true "
            f"(analysis_unit={plan.analysis_mode.unit})"
        )
    return vocabulary.load_wordlist(path) if plan.config.dictcheck.enabled else None


def write_dictcheck_outputs(
    *,
    plan: AnalysisPlan,
    artifact_plan: ArtifactPlan,
    label: str,
    counter: Mapping[str, int],
    known_words: Iterable[str] | None,
) -> KnownUnknownCounters | None:
    if not plan.config.dictcheck.enabled:
        return None
    assert known_words is not None
    split = split_known_unknown(counter, known_words)
    write_frequency_csv(artifact_plan.require(ArtifactKind.DICTCHECK_KNOWN,
                                              group=label).path,
                        split.known, header=plan.analysis_mode.csv_header)
    write_frequency_csv(artifact_plan.require(ArtifactKind.DICTCHECK_UNKNOWN,
                                              group=label).path,
                        split.unknown, header=plan.analysis_mode.csv_header)
    return split


def write_group_analysis_outputs(
    *,
    plan: AnalysisPlan,
    artifact_plan: ArtifactPlan,
    corpus: PreparedCorpus,
    counter: Counter[str],
    normalization_map: Mapping[str, str] | None,
    known_words: Iterable[str] | None,
) -> Counter[str]:
    result_counter = (
        apply_lemma_normalization(counter, normalization_map)
        if normalization_map is not None
        else counter
    )
    if plan.config.ref_tags.enabled:
        ref_tags_path = artifact_plan.require(ArtifactKind.REFERENCE_TAGS,
                                              group=corpus.label).path
        write_frequency_csv(ref_tags_path, corpus.ref_tag_counts, header=("tag", "count"))
    frequency_path = artifact_plan.require(ArtifactKind.FREQUENCY,
                                           group=corpus.label).path
    write_frequency_csv(
        frequency_path, result_counter, header=plan.analysis_mode.csv_header
    )
    write_dictcheck_outputs(
        plan=plan,
        artifact_plan=artifact_plan,
        label=corpus.label,
        counter=result_counter,
        known_words=known_words,
    )
    return result_counter
