from __future__ import annotations

import re
from collections.abc import Collection
from pathlib import Path

_ROMAN_NUMERAL_RE = re.compile(
    r"^(m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3}))$",
    re.I,
)
DEFAULT_SURFACE_ROMAN_EXCEPTIONS = frozenset({"vi", "di"})

__all__ = [
    "RomanExceptionsError",
    "effective_roman_exceptions",
    "load_roman_exceptions",
    "normalize_roman_exceptions",
    "should_drop_roman_numeral",
]


class RomanExceptionsError(ValueError):
    pass


def normalize_roman_exceptions(values: Collection[str]) -> frozenset[str]:
    return frozenset(
        normalized
        for value in values
        if (normalized := str(value).strip().lower())
    )


def load_roman_exceptions(path: Path) -> frozenset[str]:
    path = Path(path)
    if not path.exists():
        raise RomanExceptionsError(
            f"filters.roman_exceptions_file was not found: {path}"
        )
    if not path.is_file():
        raise RomanExceptionsError(
            f"filters.roman_exceptions_file must be a file: {path}"
        )
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RomanExceptionsError(
            f"Failed to read Roman numeral exceptions file: {path}"
        ) from exc
    return normalize_roman_exceptions(
        [line for line in lines if line.strip() and not line.strip().startswith("#")]
    )


def effective_roman_exceptions(
    *,
    use_lemma: bool,
    configured_exceptions: Collection[str],
) -> frozenset[str]:
    configured = normalize_roman_exceptions(configured_exceptions)
    if use_lemma:
        return configured
    return configured | DEFAULT_SURFACE_ROMAN_EXCEPTIONS


def should_drop_roman_numeral(
    key: str,
    *,
    drop_roman_numerals: bool,
    effective_exceptions: Collection[str],
) -> bool:
    normalized_key = key.strip().lower()
    exceptions = normalize_roman_exceptions(effective_exceptions)
    return (
        drop_roman_numerals
        and _ROMAN_NUMERAL_RE.fullmatch(normalized_key) is not None
        and normalized_key not in exceptions
    )
