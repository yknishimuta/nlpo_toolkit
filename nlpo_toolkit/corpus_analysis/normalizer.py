from __future__ import annotations

import re
import unicodedata

from .config import NormalizationConfig


def strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_text(text: str, config: NormalizationConfig) -> str:
    if not config.enabled:
        return text

    if config.unicode_nf:
        text = unicodedata.normalize(config.unicode_nf, text)

    if config.casefold:
        text = text.casefold()

    if config.normalize_ligatures:
        text = text.replace("æ", "ae").replace("œ", "oe")

    if config.map_u_v:
        text = text.replace("v", "u").replace("V", "U")

    if config.map_i_j:
        text = text.replace("j", "i").replace("J", "I")

    if config.strip_diacritics:
        text = strip_diacritics(text)

    text = re.sub(r'([()\[\]{}“”‘’\'"«»])', r' \1 ', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return text
