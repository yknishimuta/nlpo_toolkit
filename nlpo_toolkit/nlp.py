from __future__ import annotations
from typing import Set, Dict, List, Iterable, Optional, Mapping, Union, Callable
from collections import Counter
import unicodedata
import re
from pathlib import Path

_LIG_MAP = {
    "æ": "ae", "Æ": "ae",
    "œ": "oe", "Œ": "oe",
}
_DIACRITICS_RE = re.compile(r"[\u0300-\u036f]")

PackageType = Union[str, Mapping[str, str], None]

# Simple tokenizer
TOKEN_RE = re.compile(r"[A-Za-zĀāĒēĪīŌōŪūÆæŒœ]+")

def build_stanza_pipeline(
    lang: str = "la",
    processors: str = "tokenize,mwt,pos,lemma",
    package: PackageType = None,
    use_gpu: bool = False,
):
    """
    Build a Stanza pipeline (auto-download if missing).
    """
    if lang == "la" and package is None:
        package = "perseus"

    import stanza

    try:
        return stanza.Pipeline(
            lang=lang,
            processors=processors,
            package=package,
            use_gpu=use_gpu,
        )
    except Exception:
        stanza.download(lang, package=package)
        return stanza.Pipeline(
            lang=lang,
            processors=processors,
            package=package,
            use_gpu=use_gpu,
        )

def tokenize_all_pos(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())

def count_nouns(text: str, nlp, use_lemma: bool = True,
                upos_targets: Set[str] = frozenset({"NOUN"})) -> Counter:
    """
    Count NOUN/PROPN (configurable via upos_targets). Returns Counter of lowercase lemmas (or surface).
    """
    if not text or not text.strip():
        return Counter()
    nouns: List[str] = []
    doc = nlp(text)
    for sent in doc.sentences:
        for w in sent.words:
            if w.upos in upos_targets:
                token = (w.lemma if (use_lemma and getattr(w, "lemma", None)) else w.text)
                if token:
                    nouns.append(token.lower())
    return Counter(nouns)

def count_nouns_normalized(
    text: str,
    nlp,
    use_lemma: bool = True,
    upos_targets: Set[str] = frozenset({"NOUN"}),
    normalizer: Optional[Callable[[str], str]] = None,
):
    """
    Count NOUN tokens (optionally using lemma) and return a normalized Counter.

    - No vocabulary filtering.
    - Always returns a single Counter.
    """
    if normalizer is None:
        normalizer = normalize_token

    counter = Counter()

    if not text.strip():
        return counter

    doc = nlp(text)

    for sent in doc.sentences:
        for w in sent.words:
            if w.upos in upos_targets:
                token = (w.lemma if (use_lemma and getattr(w, "lemma", None)) else w.text)
                if token:
                    counter[normalizer(token)] += 1

    return counter

def render_stanza_package_table(
    nlp,
    requested_package: Optional[Dict[str, str]] = None,
    processors: Iterable[str] = ("tokenize", "mwt", "pos", "lemma"),
) -> List[str]:
    
    pkg_map: Dict[str, str] = {}
    if isinstance(requested_package, dict):
        for p in processors:
            if p in requested_package:
                pkg_map[p] = requested_package[p]

    try:
        cfg = getattr(nlp, "config", {})
        if isinstance(cfg, dict):
            inner = cfg.get("processors", {})
            if isinstance(inner, dict):
                for p in processors:
                    if p in inner and isinstance(inner[p], dict):
                        pkg_val = inner[p].get("package")
                        if pkg_val:
                            pkg_map[p] = pkg_val
    except Exception:
        pass

    # render table
    lines: List[str] = []
    lines.append("=== Stanza model packages ===")
    lines.append("================================")
    lines.append("| Processor | Package          |")
    lines.append("--------------------------------")
    for p in processors:
        lines.append(f"| {p:<9} | {pkg_map.get(p, '(default)'):<16} |")
    lines.append("================================")
    lines.append("")
    return lines

def iter_char_chunks(text: str, chunk_chars: int = 200_000):
    """
    Yield text in fixed-size approximate chunks, trying not to split inside a token.
    """
    N = len(text)
    i = 0
    while i < N:
        j = min(N, i + chunk_chars)

        if j < N:
            k = text.rfind(" ", i + 1, j)
            if k == -1:
                k = text.rfind("\n", i + 1, j)

            if k != -1 and k > i:
                j = k + 1

        yield text[i:j]
        i = j


def count_nouns_streaming(
    text: str,
    nlp,
    use_lemma: bool = True,
    upos_targets: Set[str] = frozenset({"NOUN"}),
    chunk_chars: int = 200_000,
    label: str = "",
):
    """
    Count NOUN lemmas (or surface forms) in streaming chunks.
    """

    total = Counter()
    if not text:
        return total

    chunks = list(iter_char_chunks(text, chunk_chars=chunk_chars))
    for k, chunk in enumerate(chunks, 1):

        # chunk 内の名詞を数える
        nouns = count_nouns(
            chunk,
            nlp,
            use_lemma=use_lemma,
            upos_targets=upos_targets,
        )
        total.update(nouns)

        if label:
            print(f"[NLP] {label}: chunk {k}/{len(chunks)} processed (chars {len(chunk):,})")

    return total

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

