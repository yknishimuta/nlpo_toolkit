from typing import Collection, Set, Optional, Iterable, Mapping, Union

from .interfaces import NLPBackend

import re
import string
import unicodedata
from pathlib import Path

_LIG_MAP = {
    "æ": "ae",
    "Æ": "ae",
    "œ": "oe",
    "Œ": "oe",
}
_DIACRITICS_RE = re.compile(r"[\u0300-\u036f]")

# stanza's package can accept a str as well as a dict specifying per-processor packages
PackageType = Union[str, Mapping[str, str], None]

_STRIP_PUNCT = string.punctuation + "“”‘’«»…—–-­"
_ROMAN_RE = re.compile(r"^(m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3}))$", re.I)
_SURFACE_ROMAN_EXCEPTIONS = frozenset({"vi", "di"}) # Surface時に保護する単語


class RomanExceptionsError(ValueError):
    pass


def load_roman_exceptions(path: Path) -> frozenset[str]:
    path = Path(path)
    if not path.exists():
        raise RomanExceptionsError(f"filters.roman_exceptions_file was not found: {path}")
    if not path.is_file():
        raise RomanExceptionsError(f"filters.roman_exceptions_file must be a file: {path}")

    items: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        raise RomanExceptionsError(
            f"Failed to read Roman numeral exceptions file: {path}"
        ) from exc

    for line in lines:
        item = line.strip().lower()
        if not item or item.startswith("#"):
            continue
        items.add(item)
    return frozenset(items)


def resolve_roman_exceptions(
    *,
    roman_exceptions: Optional[Collection[str]] = None,
    roman_exceptions_file: Optional[Path] = None,
) -> frozenset[str]:
    items = {
        str(item).strip().lower()
        for item in (roman_exceptions or ())
        if str(item).strip()
    }
    if roman_exceptions_file is not None:
        items.update(load_roman_exceptions(Path(roman_exceptions_file)))
    return frozenset(items)


def effective_roman_exceptions(
    *,
    use_lemma: bool,
    configured_exceptions: Collection[str],
) -> frozenset[str]:
    items = frozenset(str(item).strip().lower() for item in configured_exceptions if str(item).strip())
    if use_lemma:
        return items
    return items | _SURFACE_ROMAN_EXCEPTIONS


def should_drop_roman_numeral(
    key: str,
    *,
    drop_roman_numerals: bool,
    effective_exceptions: Collection[str],
) -> bool:
    return (
        drop_roman_numerals
        and _ROMAN_RE.fullmatch(key) is not None
        and key not in effective_exceptions
    )

def build_stanza_pipeline(
    lang: str = "la",
    *,
    processors: str,
    package: str = "perseus",
    use_gpu: bool = False,
) -> NLPBackend:
    """Initialize the configured Stanza backend."""
    from .backends.stanza_backend import StanzaBackend
    return StanzaBackend(
        lang=lang,
        package=package,
        use_gpu=use_gpu,
        processors=processors
    )

def build_sentence_splitter(language: str, stanza_package: str, cpu_only: bool):
    return build_stanza_pipeline(
        lang=language,
        processors="tokenize",
        package=stanza_package,
        use_gpu=not cpu_only,
    )


def iter_char_chunks(text: str, chunk_chars: int) -> Iterable[str]:
    """Splits the text based on a target character count (adjusted at whitespaces to avoid splitting words)"""
    if not isinstance(chunk_chars, int) or isinstance(chunk_chars, bool) or chunk_chars <= 0:
        raise ValueError("chunk_chars must be a positive integer")
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_chars
        if end >= text_len:
            yield text[start:]
            break
        
        # Backtrack slightly to a whitespace to avoid splitting a word
        while end > start and not text[end].isspace():
            end -= 1
        if end == start:  # Fallback in case of a single massive word
            end = start + chunk_chars
            
        yield text[start:end]
        start = end


def normalize_token(
    s: str,
    *,
    lig_map: Mapping[str, str] = _LIG_MAP,
    strip_diacritics: bool = True,
    lower: bool = True,
) -> str:
    """
    Generic normalizer for vocab lookup:
    - NFKD normalize
    - optionally remove diacritics
    - optionally expand ligatures via lig_map
    - optionally lowercase
    """
    if not s:
        return ""
    t = unicodedata.normalize("NFKD", s)
    if strip_diacritics:
        t = _DIACRITICS_RE.sub("", t)
    if lig_map:
        t = "".join(lig_map.get(ch, ch) for ch in t)
    if lower:
        t = t.lower()
    return t


def load_vocab(path: Path) -> Set[str]:
    """
    Load lemmas/words from a UTF-8 text file (one item per line).
    The file should already be normalized to the same convention
    (lowercased, ligatures expanded), or we normalize when checking.
    """
    vocab = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        w = line.strip()
        if not w or w.startswith("#"):
            continue
        vocab.add(w)
    return vocab
