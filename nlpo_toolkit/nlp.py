from collections import Counter
from typing import Set, Optional, Counter as CounterType, Dict, List, Iterable, Mapping, Union, Callable, Any

from .models import NLPDocument, NLPToken
from .interfaces import NLPBackend

from collections import Counter
import unicodedata
import re
import csv
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

# Simple tokenizer
TOKEN_RE = re.compile(r"[A-Za-zĀāĒēĪīŌōŪūÆæŒœ]+")


def build_stanza_pipeline(
    lang: str = "la",
    processors: str = "tokenize,mwt,pos,lemma",
    package: str = "perseus",
    use_gpu: bool = False,
) -> NLPBackend:
    """
    Function to maintain compatibility.
    Initializes and returns a StanzaBackend internally.
    """
    from .backends.stanza_backend import StanzaBackend
    return StanzaBackend(
        lang=lang,
        package=package,
        use_gpu=use_gpu,
        processors=processors
    )

def count_nouns(
    text: str,
    nlp: NLPBackend,
    use_lemma: bool = True,
    upos_targets: Optional[Set[str]] = None,
    ref_tag_detector: Optional[Callable[[str], str]] = None,
    ref_tag_counter: Optional[CounterType[str]] = None,
) -> CounterType[str]:
    """
    Analyzes text using the NLP backend and counts the occurrences of the specified parts of speech.
    """
    if upos_targets is None:
        upos_targets = {"NOUN"}

    doc: NLPDocument = nlp(text)
    counts = Counter()

    for sent in doc.sentences:
        for token in sent.tokens:
            # Count specified parts of speech (e.g., NOUN)
            if token.upos in upos_targets:
                key = token.lemma.strip().lower() if (use_lemma and token.lemma) else token.text.strip().lower()
                if key:
                    counts[key] += 1
            
            # Detect and count reference tags (REF)
            if ref_tag_detector is not None:
                ref_tag = ref_tag_detector(token.text)
                if ref_tag and ref_tag_counter is not None:
                    ref_tag_counter[ref_tag] += 1

    return counts

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

def _iter_sentence_words_with_offsets(sent):
    """
    Yield (word_like, token_start_char_in_chunk).

    Priority:
    - sent.tokens -> token.start_char + token.words
    - sent.words  -> word.start_char (if present), else None

    This is needed for trace offsets tests.
    """
    # Prefer tokens if available (keeps token-level start_char)
    for tok in getattr(sent, "tokens", []) or []:
        start_char = getattr(tok, "start_char", None)
        twords = getattr(tok, "words", None)
        if twords:
            for w in twords:
                yield w, start_char
        else:
            # token itself might be used as a fallback word-like object
            yield tok, start_char

    # If no tokens, fall back to sent.words
    words = getattr(sent, "words", None)
    if words:
        for w in words:
            yield w, getattr(w, "start_char", None)

def count_nouns_doc(
    doc,
    use_lemma: bool = True,
    upos_targets: Set[str] = frozenset({"NOUN"}),
    *,
    ref_tag_detector: Optional[Callable[[str], str]] = None,
    ref_tag_counter: Optional[Counter] = None,
) -> Counter:
    counter = Counter()

    for sent in getattr(doc, "sentences", []) or []:
        for w in _iter_sentence_words(sent):
            upos = getattr(w, "upos", None)
            if upos not in upos_targets:
                continue

            lemma = getattr(w, "lemma", None)
            text_form = getattr(w, "text", None)
            token = lemma if (use_lemma and lemma) else text_form
            if not token:
                continue

            key = token.strip().lower()
            if not key:
                continue

            if ref_tag_detector is not None:
                tag = ref_tag_detector(key)
                if tag:
                    if ref_tag_counter is not None:
                        ref_tag_counter[tag] += 1
                    continue

            counter[key] += 1

    return counter


def count_nouns(
    text: str,
    nlp,
    use_lemma: bool = True,
    upos_targets: Set[str] = frozenset({"NOUN"}),
    *,
    ref_tag_detector: Optional[Callable[[str], str]] = None,
    ref_tag_counter: Optional[Counter] = None,
) -> Counter:
    if not text or not text.strip():
        return Counter()
    doc = nlp(text)
    return count_nouns_doc(
        doc,
        use_lemma=use_lemma,
        upos_targets=upos_targets,
        ref_tag_detector=ref_tag_detector,
        ref_tag_counter=ref_tag_counter,
    )


def count_nouns_normalized(
    text: str,
    nlp,
    use_lemma: bool = True,
    upos_targets: Set[str] = frozenset({"NOUN"}),
    normalizer: Optional[Callable[[str], str]] = None,
):
    if normalizer is None:
        normalizer = normalize_token

    counter = Counter()
    if not text.strip():
        return counter

    doc = nlp(text)

    for sent in getattr(doc, "sentences", []) or []:
        for w in _iter_sentence_words(sent):
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

def iter_char_chunks(text: str, chunk_chars: int) -> Iterable[str]:
    """Splits the text based on a target character count (adjusted at whitespaces to avoid splitting words)"""
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

def count_nouns_streaming(
    text: str,
    nlp: NLPBackend,
    use_lemma: bool = True,
    upos_targets: Optional[Set[str]] = None,
    chunk_chars: int = 200_000,
    label: str = "",
    ref_tag_detector: Optional[Callable[[str], str]] = None,
    ref_tag_counter: Optional[CounterType[str]] = None,
    trace_tsv: Optional[Path] = None,
    trace_max_rows: int = 0,
    trace_only_keys: Optional[Set[str]] = None,
    trace_write_truncation_marker: bool = True,
) -> CounterType[str]:
    """
    Performs streaming NLP analysis and counting while splitting the text into chunks.
    """
    if upos_targets is None:
        upos_targets = {"NOUN"}

    # If trace_tsv is specified, delegate to the internal function with detailed trace output
    if trace_tsv is not None:
        return _count_nouns_streaming_trace(
            text=text,
            nlp=nlp,
            use_lemma=use_lemma,
            upos_targets=upos_targets,
            chunk_chars=chunk_chars,
            label=label,
            trace_tsv=trace_tsv,
            trace_max_rows=trace_max_rows,
            trace_only_keys=trace_only_keys,
            trace_write_truncation_marker=trace_write_truncation_marker,
            ref_tag_detector=ref_tag_detector,
            ref_tag_counter=ref_tag_counter,
        )

    counts = Counter()
    for chunk in iter_char_chunks(text, chunk_chars):
        doc = nlp(chunk)  # Returns the common data model (NLPDocument)
        
        for sent in doc.sentences:
            for token in sent.tokens:
                if token.upos in upos_targets:
                    key = token.lemma.strip().lower() if (use_lemma and token.lemma) else token.text.strip().lower()
                    if key:
                        counts[key] += 1
                        
    return counts

def _count_nouns_streaming_trace(
    text: str,
    nlp: NLPBackend,
    use_lemma: bool,
    upos_targets: Set[str],
    chunk_chars: int,
    label: str,
    trace_tsv: Path,
    trace_max_rows: int,
    trace_only_keys: Optional[Set[str]],
    trace_write_truncation_marker: bool,
    ref_tag_detector: Optional[Callable[[str], str]],
    ref_tag_counter: Optional[CounterType[str]],
) -> CounterType[str]:
    """Streaming counting process with trace (TSV) output"""
    counts = Counter()
    
    trace_tsv = Path(trace_tsv)
    trace_tsv.parent.mkdir(parents=True, exist_ok=True)
    
    if trace_only_keys is not None:
        trace_only_keys = {k.lower() for k in trace_only_keys}

    with trace_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "label", "chunk", "sent_idx", "token_idx",
            "token_char_start_in_chunk", "token_char_start_in_text",
            "sentence", "token", "lemma", "upos", "ref_tag", "global_row"
        ])  #

        global_row = 0
        chunk_base_offset = 0
        truncated = False

        for chunk_idx, chunk in enumerate(iter_char_chunks(text, chunk_chars)):
            doc = nlp(chunk)
            
            for sent_idx, sent in enumerate(doc.sentences):
                # Full sentence text (if not retained, concatenate tokens)
                sent_text_str = sent.text if sent.text else " ".join([t.text for t in sent.tokens])
                
                for token_idx, token in enumerate(sent.tokens):
                    if token.upos in upos_targets:
                        key = token.lemma.strip().lower() if (use_lemma and token.lemma) else token.text.strip().lower()
                        if not key:
                            continue
                            
                        counts[key] += 1
                        
                        # ------------------------------------------------
                        # Trace write check
                        # ------------------------------------------------
                        if truncated:
                            continue
                            
                        if trace_only_keys and key not in trace_only_keys:
                            continue
                            
                        if trace_max_rows > 0 and global_row >= trace_max_rows:
                            if trace_write_truncation_marker:
                                writer.writerow([
                                    label, chunk_idx, sent_idx, token_idx,
                                    token.start_char, chunk_base_offset + token.start_char,
                                    sent_text_str, "(trace stopped; counting continues)",
                                    "", "TRACE_TRUNCATED", "", global_row + 1
                                ])  #
                            truncated = True
                            continue

                        # Detect reference tags
                        ref_tag = ""
                        if ref_tag_detector:
                            ref_tag = ref_tag_detector(token.text)
                            if ref_tag and ref_tag_counter is not None:
                                ref_tag_counter[ref_tag] += 1

                        # Write to TSV
                        writer.writerow([
                            label,
                            chunk_idx,
                            sent_idx,
                            token_idx,
                            token.start_char,
                            chunk_base_offset + token.start_char,
                            sent_text_str,
                            token.text,
                            token.lemma or "",
                            token.upos,
                            ref_tag,
                            global_row + 1
                        ])
                        global_row += 1
                        
            # Update absolute position offset for the next chunk
            chunk_base_offset += len(chunk)

    return counts

def _count_nouns_streaming_trace(
    text: str,
    nlp: NLPBackend,
    use_lemma: bool,
    upos_targets: Set[str],
    chunk_chars: int,
    label: str,
    trace_tsv: Path,
    trace_max_rows: int,
    trace_only_keys: Optional[Set[str]],
    trace_write_truncation_marker: bool,
    ref_tag_detector: Optional[Callable[[str], str]],
    ref_tag_counter: Optional[CounterType[str]],
) -> CounterType[str]:
    """Streaming counting process with trace (TSV) output"""
    counts = Counter()
    
    trace_tsv = Path(trace_tsv)
    trace_tsv.parent.mkdir(parents=True, exist_ok=True)
    
    if trace_only_keys is not None:
        trace_only_keys = {k.lower() for k in trace_only_keys}

    with trace_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "label", "chunk", "sent_idx", "token_idx",
            "token_char_start_in_chunk", "token_char_start_in_text",
            "sentence", "token", "lemma", "upos", "ref_tag", "global_row"
        ])  #

        global_row = 0
        chunk_base_offset = 0
        truncated = False

        for chunk_idx, chunk in enumerate(iter_char_chunks(text, chunk_chars)):
            doc = nlp(chunk)
            
            for sent_idx, sent in enumerate(doc.sentences):
                # Full sentence text (if not retained, concatenate tokens)
                sent_text_str = sent.text if sent.text else " ".join([t.text for t in sent.tokens])
                
                for token_idx, token in enumerate(sent.tokens):
                    if token.upos in upos_targets:
                        key = token.lemma.strip().lower() if (use_lemma and token.lemma) else token.text.strip().lower()
                        if not key:
                            continue
                            
                        counts[key] += 1
                        
                        # ------------------------------------------------
                        # Trace write check
                        # ------------------------------------------------
                        if truncated:
                            continue
                            
                        if trace_only_keys and key not in trace_only_keys:
                            continue
                            
                        if trace_max_rows > 0 and global_row >= trace_max_rows:
                            if trace_write_truncation_marker:
                                writer.writerow([
                                    label, chunk_idx, sent_idx, token_idx,
                                    token.start_char, chunk_base_offset + token.start_char,
                                    sent_text_str, "(trace stopped; counting continues)",
                                    "", "TRACE_TRUNCATED", "", global_row + 1
                                ])  #
                            truncated = True
                            continue

                        # Detect reference tags
                        ref_tag = ""
                        if ref_tag_detector:
                            ref_tag = ref_tag_detector(token.text)
                            if ref_tag and ref_tag_counter is not None:
                                ref_tag_counter[ref_tag] += 1

                        # Write to TSV
                        writer.writerow([
                            label,
                            chunk_idx,
                            sent_idx,
                            token_idx,
                            token.start_char,
                            chunk_base_offset + token.start_char,
                            sent_text_str,
                            token.text,
                            token.lemma or "",
                            token.upos,
                            ref_tag,
                            global_row + 1
                        ])
                        global_row += 1
                        
            # Update absolute position offset for the next chunk
            chunk_base_offset += len(chunk)

    return counts

def _count_nouns_streaming_fast(
    text: str,
    nlp: NLPBackend,  # <- ★ Changed this to the common interface
    use_lemma: bool = True,
    upos_targets: Set[str] = frozenset({"NOUN"}),
    chunk_chars: int = 200_000,
    label: str = "",
    *,
    ref_tag_detector: Optional[Callable[[str], str]] = None,
    ref_tag_counter: Optional[CounterType[str]] = None,
) -> CounterType[str]:
    """
    Fast path: streaming without materializing chunk list (memory-friendly).
    """
    total = Counter()
    if not text:
        return total

    for k, chunk in enumerate(iter_char_chunks(text, chunk_chars=chunk_chars), 1):
        nouns = count_nouns(
            chunk,
            nlp,
            use_lemma=use_lemma,
            upos_targets=upos_targets,
            ref_tag_detector=ref_tag_detector,
            ref_tag_counter=ref_tag_counter,
        )
        total.update(nouns)

        if label:
            print(f"[NLP] {label}: chunk {k} processed (chars {len(chunk):,})")

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
    trace_only_keys: Optional[Set[str]] = None,
    trace_write_truncation_marker: bool = True,
    ref_tag_detector: Optional[Callable[[str], str]] = None,
    ref_tag_counter: Optional[Counter] = None,
) -> Counter:
    """
    Public API:
      - trace_tsv is None => fast path
      - trace_tsv given   => trace path

    ref_tag_detector: key -> ref_tag label (non-empty) or "" (not a ref_tag).
    ref_tag_counter:  mutable Counter; ref_tag hits are accumulated here.
    """
    if trace_tsv is None:
        return _count_nouns_streaming_fast(
            text,
            nlp,
            use_lemma=use_lemma,
            upos_targets=upos_targets,
            chunk_chars=chunk_chars,
            label=label,
            ref_tag_detector=ref_tag_detector,
            ref_tag_counter=ref_tag_counter,
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
        trace_write_truncation_marker=trace_write_truncation_marker,
        trace_only_keys=trace_only_keys,
        ref_tag_detector=ref_tag_detector,
        ref_tag_counter=ref_tag_counter,
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