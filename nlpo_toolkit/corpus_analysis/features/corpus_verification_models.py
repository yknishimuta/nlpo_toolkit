from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nlpo_toolkit.stylometry.models import InputFormat
from nlpo_toolkit.stylometry.verification_models import VerificationThresholdSettings
from nlpo_toolkit.stylometry.verification_results import VerificationResult

from ..requests import CorpusPreparationRequest
from .character_ngrams import CharacterNgramTerm
from .models import FeatureRequest
from .morphology import MorphologyVocabulary
from .upos_ngrams import UposNgramTerm
from .vocabulary_audit import feature_vocabulary_sha256


@dataclass(frozen=True)
class CorpusVerificationRequest:
    features: FeatureRequest
    metadata_path: Path
    candidate_author: str
    query_work: str
    thresholds: VerificationThresholdSettings = VerificationThresholdSettings()
    metadata_format: InputFormat = "csv"
    metadata_group_column: str = "group"
    author_column: str = "author"
    work_column: str = "work"

    def __post_init__(self) -> None:
        if not isinstance(self.metadata_path, Path):
            raise ValueError("metadata path must be a Path")
        if self.metadata_format not in ("csv", "tsv"):
            raise ValueError("metadata format must be 'csv' or 'tsv'")
        for value, name in (
            (self.candidate_author, "candidate author"),
            (self.query_work, "query work"),
            (self.metadata_group_column, "metadata group column"),
            (self.author_column, "author column"),
            (self.work_column, "work column"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")

    @property
    def corpus(self) -> CorpusPreparationRequest:
        return self.features.corpus


@dataclass(frozen=True)
class CorpusVerificationVocabularyAudit:
    query_work: str
    mfw_terms: tuple[str, ...]
    character_ngrams: tuple[CharacterNgramTerm, ...]
    upos_ngrams: tuple[UposNgramTerm, ...]
    morphology: MorphologyVocabulary | None = None
    fit_scope: str = "reference_works_only"

    def __post_init__(self) -> None:
        if not self.query_work.strip():
            raise ValueError("query work must not be empty")
        if self.fit_scope != "reference_works_only":
            raise ValueError("corpus verification vocabulary fit scope is invalid")
        object.__setattr__(self, "mfw_terms", tuple(self.mfw_terms))
        object.__setattr__(self, "character_ngrams", tuple(self.character_ngrams))
        object.__setattr__(self, "upos_ngrams", tuple(self.upos_ngrams))

    @property
    def sha256(self) -> str:
        return feature_vocabulary_sha256(
            mfw_terms=self.mfw_terms,
            character_ngrams=self.character_ngrams,
            upos_ngrams=self.upos_ngrams,
            morphology=self.morphology,
        )

    @property
    def selected_mfw_count(self) -> int:
        return len(self.mfw_terms)

    @property
    def selected_character_ngram_count(self) -> int:
        return len(self.character_ngrams)

    @property
    def selected_upos_ngram_count(self) -> int:
        return len(self.upos_ngrams)

    @property
    def selected_morphology_count(self) -> int:
        return 0 if self.morphology is None else len(self.morphology.values) + len(
            self.morphology.bundles
        )


@dataclass(frozen=True)
class CorpusVerificationResult:
    verification: VerificationResult
    selected_feature_count: int
    vocabulary: CorpusVerificationVocabularyAudit

    def __post_init__(self) -> None:
        if self.selected_feature_count < 1:
            raise ValueError("selected feature count must be positive")
