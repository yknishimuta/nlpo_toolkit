from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from nlpo_toolkit.stylometry.evaluation_results import (
    LeaveOneWorkOutFoldResult,
    LeaveOneWorkOutSummary,
)
from nlpo_toolkit.stylometry.models import InputFormat

from ..requests import CorpusPreparationRequest
from .character_ngrams import CharacterNgramTerm
from .models import FeatureRequest
from .upos_ngrams import UposNgramTerm
from .morphology import MorphologyVocabulary


@dataclass(frozen=True)
class CorpusLowoRequest:
    features: FeatureRequest
    metadata_path: Path
    metadata_format: InputFormat = "csv"
    metadata_group_column: str = "group"
    author_column: str = "author"
    work_column: str = "work"

    def __post_init__(self) -> None:
        if self.metadata_format not in ("csv", "tsv"):
            raise ValueError("metadata format must be 'csv' or 'tsv'")
        for value, name in (
            (self.metadata_group_column, "metadata group column"),
            (self.author_column, "author column"),
            (self.work_column, "work column"),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")

    @property
    def corpus(self) -> CorpusPreparationRequest:
        return self.features.corpus


@dataclass(frozen=True)
class FoldVocabularyAudit:
    fold_index: int
    test_work: str
    mfw_terms: tuple[str, ...]
    character_ngrams: tuple[CharacterNgramTerm, ...]
    upos_ngrams: tuple[UposNgramTerm, ...]
    morphology: MorphologyVocabulary | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mfw_terms", tuple(self.mfw_terms))
        object.__setattr__(self, "character_ngrams", tuple(self.character_ngrams))
        object.__setattr__(self, "upos_ngrams", tuple(self.upos_ngrams))

    @property
    def sha256(self) -> str:
        parts = [f"mfw:{len(self.mfw_terms)}"]
        parts.extend(f"m:{len(term)}:{term}" for term in self.mfw_terms)
        parts.extend(
            f"c:{term.mode.value}:{term.size}:{len(term.value)}:"
            f"{term.value}:{term.column_name}"
            for term in self.character_ngrams
        )
        if self.morphology is not None:
            parts.extend(f"ma:{attribute}" for attribute in self.morphology.attributes)
            parts.extend(
                f"mv:{item.attribute}={item.value}" for item in self.morphology.values
            )
            parts.extend(
                "mb:"
                + "|".join(f"{item.attribute}={item.value}" for item in bundle.features)
                for bundle in self.morphology.bundles
            )
        parts.extend(
            f"u:{term.size}:{'|'.join(term.tags)}:{term.column_name}"
            for term in self.upos_ngrams
        )
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CorpusLowoFoldResult:
    evaluation: LeaveOneWorkOutFoldResult
    selected_feature_count: int
    vocabulary: FoldVocabularyAudit

    @property
    def selected_mfw_count(self) -> int:
        return len(self.vocabulary.mfw_terms)

    @property
    def selected_character_ngram_count(self) -> int:
        return len(self.vocabulary.character_ngrams)

    @property
    def selected_upos_ngram_count(self) -> int:
        return len(self.vocabulary.upos_ngrams)

    @property
    def selected_morphology_count(self) -> int:
        morphology = self.vocabulary.morphology
        return (
            0
            if morphology is None
            else len(morphology.values) + len(morphology.bundles)
        )


@dataclass(frozen=True)
class CorpusLowoResult:
    folds: tuple[CorpusLowoFoldResult, ...]
    summary: LeaveOneWorkOutSummary

    def __post_init__(self) -> None:
        object.__setattr__(self, "folds", tuple(self.folds))
