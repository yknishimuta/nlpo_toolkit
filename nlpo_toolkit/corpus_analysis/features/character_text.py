from __future__ import annotations

import re

from .errors import FeatureError
from .models import AnalyzedFeatureCorpus


def normalize_character_stream(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


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
