from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

DEFAULT_LIGATURE_MAP: Mapping[str, str] = {
    "æ": "ae",
    "Æ": "ae",
    "œ": "oe",
    "Œ": "oe",
}
_COMBINING_DIACRITICS_RE = re.compile(r"[\u0300-\u036f]")

__all__ = ["DEFAULT_LIGATURE_MAP", "normalize_token"]


def normalize_token(
    value: str,
    *,
    ligature_map: Mapping[str, str] = DEFAULT_LIGATURE_MAP,
    strip_diacritics: bool = True,
    lower: bool = True,
) -> str:
    """Normalize a token for vocabulary lookup."""
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    if strip_diacritics:
        normalized = _COMBINING_DIACRITICS_RE.sub("", normalized)
    if ligature_map:
        normalized = "".join(ligature_map.get(char, char) for char in normalized)
    return normalized.lower() if lower else normalized
