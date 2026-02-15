from __future__ import annotations
from typing import Set, Dict, List, Iterable, Optional, Mapping, Union, Callable
from collections import Counter
import unicodedata, re, csv
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

def build_sentence_splitter(language: str, stanza_package: str, cpu_only: bool):
    return build_stanza_pipeline(
        lang=language,
        processors="tokenize",
        package=stanza_package,
        use_gpu=not cpu_only,
    )

def tokenize_all_pos(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())

def _iter_sentence_words(sent):
    words = getattr(sent, "words", None)
    if words:
        for w in words:
            yield w
        return

    # fallback: sent.tokens -> token.words
    for tok in getattr(sent, "tokens", []) or []:
        twords = getattr(tok, "words", None)
        if twords:
            for w in twords:
                yield w
        else:
            yield tok

def count_nouns(doc):
    """
    Count NOUN lemmas in a Stanza document.
    Safe against words/tokens structure differences.
    """
    noun_counts = {}

    for sent in doc.sentences:
        # 1) Prefer words
        if getattr(sent, "words", None):
            iterable = sent.words

        # 2) Fallback: tokens -> token.words
        elif getattr(sent, "tokens", None):
            iterable = []
            for tok in sent.tokens:
                if getattr(tok, "words", None):
                    iterable.extend(tok.words)
                else:
                    iterable.append(tok)
        else:
            continue

        for w in iterable:
            upos = getattr(w, "upos", None)
            lemma = getattr(w, "lemma", None)

            if upos != "NOUN":
                continue
            if not lemma:
                continue

            noun_counts[lemma] = noun_counts.get(lemma, 0) + 1

    return noun_counts

def count_nouns_normalized(
    text: str,
    nlp,
    use_lemma: bool = True,
    upos_targets: Set[str] = frozenset({"NOUN"}),
    normalizer: Optional[Callable[[str], str]] = None,
):
    """
    Count NOUN tokens (optionally using lemma) and return a normalized Counter.

    - Safe against words/tokens structure differences
    - No vocabulary filtering
    - Always returns a single Counter
    """
    if normalizer is None:
        normalizer = normalize_token

    counter = Counter()

    if not text.strip():
        return counter

    doc = nlp(text)

    for sent in doc.sentences:

        # ---- words 優先 ----
        if getattr(sent, "words", None):
            iterable = sent.words

        # ---- fallback: tokens -> token.words ----
        elif getattr(sent, "tokens", None):
            iterable = []
            for tok in sent.tokens:
                if getattr(tok, "words", None):
                    iterable.extend(tok.words)
                else:
                    iterable.append(tok)
        else:
            continue

        for w in iterable:
            upos = getattr(w, "upos", None)
            if upos not in upos_targets:
                continue

            lemma = getattr(w, "lemma", None)
            text_form = getattr(w, "text", None)

            token = lemma if (use_lemma and lemma) else text_form

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

def _count_nouns_streaming_fast(
    text: str,
    nlp,
    use_lemma: bool = True,
    upos_targets: Set[str] = frozenset({"NOUN"}),
    chunk_chars: int = 200_000,
    label: str = "",
) -> Counter:
    """
    Fast path: existing behavior (no tracing).
    """
    total = Counter()
    if not text:
        return total

    chunks = list(iter_char_chunks(text, chunk_chars=chunk_chars))
    for k, chunk in enumerate(chunks, 1):
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


def _count_nouns_streaming_trace(
    text: str,
    nlp,
    use_lemma: bool = True,
    upos_targets: Set[str] = frozenset({"NOUN"}),
    chunk_chars: int = 200_000,
    label: str = "",
    *,
    trace_tsv: Path,
    trace_max_rows: int = 0,  # 0 => unlimited
) -> Counter:
    """
    Trace path: dump evidence rows to TSV while counting nouns.

    TSV columns:
      chunk_idx, sentence_text, token_text, lemma, upos
    """
    total = Counter()
    if not text:
        return total

    trace_tsv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with trace_tsv.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp, delimiter="\t")
        w.writerow(["chunk", "sentence", "token", "lemma", "upos"])

        chunks = list(iter_char_chunks(text, chunk_chars=chunk_chars))
        for k, chunk in enumerate(chunks, 1):
            doc = nlp(chunk)

            for sent in getattr(doc, "sentences", []):
                sent_text = getattr(sent, "text", "") or ""

                for wd in _iter_sentence_words(sent):
                    upos = getattr(wd, "upos", "") or ""
                    if upos not in upos_targets:
                        continue

                    token = getattr(wd, "text", "") or ""
                    lemma = getattr(wd, "lemma", "") or ""

                    key = lemma if (use_lemma and lemma) else token
                    key = (key or "").strip().lower()
                    if key:
                        total[key] += 1

                    if trace_max_rows and rows_written >= trace_max_rows:
                        continue
                    w.writerow([k, sent_text, token, lemma, upos])
                    rows_written += 1

            if label:
                print(f"[NLP] {label}: chunk {k}/{len(chunks)} processed (chars {len(chunk):,})")

    return total


def count_nouns_streaming(
    text: str,
    nlp,
    use_lemma: bool = True,
    upos_targets: Set[str] = frozenset({"NOUN"}),
    chunk_chars: int = 200_000,
    label: str = "",
    *,
    trace_tsv: Optional[Path] = None,
    trace_max_rows: int = 0,
) -> Counter:
    """
    Public API:
      - trace_tsv is None => fast path
      - trace_tsv given   => trace path
    """
    if trace_tsv is None:
        return _count_nouns_streaming_fast(
            text,
            nlp,
            use_lemma=use_lemma,
            upos_targets=upos_targets,
            chunk_chars=chunk_chars,
            label=label,
        )

    return _count_nouns_streaming_trace(
        text,
        nlp,
        use_lemma=use_lemma,
        upos_targets=upos_targets,
        chunk_chars=chunk_chars,
        label=label,
        trace_tsv=trace_tsv,
        trace_max_rows=trace_max_rows,
    )

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

