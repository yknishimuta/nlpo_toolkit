from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from .errors import CleanerLexiconError


LexiconMap = Mapping[str, str]
EMPTY_LEXICON_MAP: LexiconMap = MappingProxyType({})


def load_lexicon_map(path: str | Path) -> LexiconMap:
    source = Path(path).resolve()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CleanerLexiconError(f"Failed to read lexicon map {source}: {exc}") from exc
    mapping: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("\t")
        if len(parts) < 2:
            raise CleanerLexiconError(f"{source}:{line_number}: expected TSV 'from\\tto'")
        key = parts[0].strip()
        if key:
            mapping[key] = parts[1].strip()
    return MappingProxyType(mapping)


def apply_lexicon_map(text: str, mapping: Mapping[str, str]) -> str:
    if not text or not mapping:
        return text
    keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(key) for key in keys) + r")\b")
    return pattern.sub(lambda match: mapping.get(match.group(1), match.group(1)), text)
