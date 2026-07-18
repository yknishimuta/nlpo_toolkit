from __future__ import annotations

from dataclasses import dataclass

from .character_ngrams import CharacterNgramVocabulary
from .upos_ngrams import UposNgramVocabulary
from .morphology import MorphologyVocabulary


@dataclass(frozen=True)
class FeatureVocabulary:
    mfw_terms: tuple[str, ...] = ()
    character_ngrams: CharacterNgramVocabulary | None = None
    upos_ngrams: UposNgramVocabulary | None = None
    morphology: MorphologyVocabulary | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mfw_terms", tuple(self.mfw_terms))
