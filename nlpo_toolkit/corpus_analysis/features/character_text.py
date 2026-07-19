from __future__ import annotations

import re
import unicodedata

from .errors import FeatureError
from .models import AnalyzedFeatureCorpus, CharacterNgramMode


def normalize_character_stream(
    text: str, *, mode: CharacterNgramMode = CharacterNgramMode.FULL
) -> str:
    if not isinstance(mode, CharacterNgramMode):
        raise FeatureError("character n-gram mode must be CharacterNgramMode")
    lowered = text.lower()
    if mode is CharacterNgramMode.FULL:
        return re.sub(r"\s+", " ", lowered).strip()
    if mode is CharacterNgramMode.LETTERS_ONLY:
        return "".join(
            character
            for character in lowered
            if unicodedata.category(character).startswith(("L", "M"))
        )
    normalized = []
    for character in lowered:
        category = unicodedata.category(character)
        keep = (
            not category.startswith("P")
            if mode is CharacterNgramMode.NO_PUNCTUATION
            else category.startswith(("L", "M"))
        )
        normalized.append(character if keep and not character.isspace() else " ")
    return re.sub(r" +", " ", "".join(normalized)).strip()


def encode_character_ngram(value: str) -> str:
    parts = []
    for character in value:
        if "a" <= character <= "z" or "0" <= character <= "9":
            parts.append(character)
        elif character == " ":
            parts.append("_sp_")
        else:
            parts.append(f"_u{ord(character):06x}_")
    return "".join(parts)


def feature_unit_character_text(corpus: AnalyzedFeatureCorpus) -> str:
    if corpus.text is not None:
        return corpus.text
    sample_id = (
        corpus.sample.sample_id if corpus.sample is not None else corpus.source.label
    )
    raise FeatureError(
        "character n-gram features require exact text offsets; "
        f"sample {sample_id!r} has no usable character span"
    )
