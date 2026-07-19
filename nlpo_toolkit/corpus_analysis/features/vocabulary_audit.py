from __future__ import annotations

import hashlib

from .character_ngrams import CharacterNgramTerm
from .morphology import MorphologyVocabulary
from .upos_ngrams import UposNgramTerm


def feature_vocabulary_sha256(
    *,
    mfw_terms: tuple[str, ...],
    character_ngrams: tuple[CharacterNgramTerm, ...],
    upos_ngrams: tuple[UposNgramTerm, ...],
    morphology: MorphologyVocabulary | None,
) -> str:
    parts = [f"mfw:{len(mfw_terms)}"]
    parts.extend(f"m:{len(term)}:{term}" for term in mfw_terms)
    parts.extend(
        f"c:{term.mode.value}:{term.size}:{len(term.value)}:"
        f"{term.value}:{term.column_name}"
        for term in character_ngrams
    )
    if morphology is not None:
        parts.extend(f"ma:{attribute}" for attribute in morphology.attributes)
        parts.extend(f"mv:{item.attribute}={item.value}" for item in morphology.values)
        parts.extend(
            "mb:"
            + "|".join(f"{item.attribute}={item.value}" for item in bundle.features)
            for bundle in morphology.bundles
        )
    parts.extend(
        f"u:{term.size}:{'|'.join(term.tags)}:{term.column_name}"
        for term in upos_ngrams
    )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
