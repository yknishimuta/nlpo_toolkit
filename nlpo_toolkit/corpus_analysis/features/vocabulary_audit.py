from __future__ import annotations

import hashlib

from .character_ngrams import CharacterNgramTerm
from .content_masked_character_ngrams import ContentMaskedCharacterNgramTerm
from .content_masking import CONTENT_MASK_POLICY_ID
from .morphology import MorphologyVocabulary
from .upos_ngrams import UposNgramTerm


def feature_vocabulary_sha256(
    *,
    mfw_terms: tuple[str, ...],
    character_ngrams: tuple[CharacterNgramTerm, ...],
    content_masked_character_ngrams: tuple[ContentMaskedCharacterNgramTerm, ...] = (),
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
    if content_masked_character_ngrams:
        parts.append(f"cm-policy:{CONTENT_MASK_POLICY_ID}")
        parts.extend(
            f"cm:{term.size}:{len(term.value)}:{term.value}:{term.column_name}"
            for term in content_masked_character_ngrams
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
