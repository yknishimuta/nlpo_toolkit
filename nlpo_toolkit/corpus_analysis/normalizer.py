from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping


def strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _normalization_value(cfg: object, key: str, default: Any = None) -> Any:
    if hasattr(cfg, "normalization"):
        norm_cfg = getattr(cfg, "normalization")
        return getattr(norm_cfg, key, default)
    if isinstance(cfg, Mapping):
        norm_cfg = cfg.get("normalization", {})
        if isinstance(norm_cfg, Mapping):
            return norm_cfg.get(key, default)
    return default


def normalize_text(text: str, cfg: object) -> str:
    if _normalization_value(cfg, "enabled", True) is False:
        return text

    # Unicode normalization
    nf = _normalization_value(cfg, "unicode_nf", None)
    if nf:
        text = unicodedata.normalize(nf, text)

    # casefold
    if _normalization_value(cfg, "casefold", False):
        text = text.casefold()

    # ligatures
    if _normalization_value(cfg, "normalize_ligatures", False):
        text = text.replace("æ", "ae").replace("œ", "oe")

    # u/v
    if _normalization_value(cfg, "map_u_v", False):
        text = text.replace("v", "u").replace("V", "U")

    # i/j
    if _normalization_value(cfg, "map_i_j", False):
        text = text.replace("j", "i").replace("J", "I")

    # diacritics
    if _normalization_value(cfg, "strip_diacritics", False):
        text = strip_diacritics(text)

    text = re.sub(r'([()\[\]{}“”‘’\'"«»])', r' \1 ', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return text
