from __future__ import annotations

from typing import (
    Set,
    Dict,
    List,
    Iterable,
    Optional,
    Mapping,
    Union,
    Callable,
    Any,
)
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

# stanza の package は str のほか、processorごとの dict 指定も受け得る
PackageType = Union[str, Mapping[str, str], None]

# Simple tokenizer
TOKEN_RE = re.compile(r"[A-Za-zĀāĒēĪīŌōŪūÆæŒœ]+")


def build_stanza_pipeline(
    lang: str = "la",
    processors: str = "tokenize,mwt,pos,lemma",
    package: PackageType = None,
    use_gpu: bool = False,
    *,
    # ---- backwards/compat knobs ----
    stanza_package: PackageType = None,
    # download挙動を明示的に制御したいケースのため（デフォルトは「従来通り auto-download」）
    auto_download: bool = True,
    # stanza リソースディレクトリ等を渡したい場合のため
    stanza_dir: Optional[str] = None,
    # stanza.Pipeline の download_method を使いたい場合（例: "reuse_resources", "download_resources" 等）
    download_method: Optional[str] = None,
    # 将来拡張/呼び出し側の追加引数用
    **kwargs: Any,
):
    """
    Build a Stanza pipeline (auto-download if missing).

    Backward-compat:
    - `package` 引数を維持（既存実装と同じ）
    - 追加で `stanza_package` でも指定できる（他モジュール/設定との整合用）
    - 既定では従来通り auto-download する（auto_download=True）

    Notes:
    - stanza の import は関数内にして pytest の collection 時の副作用を最小化
    """

    # 優先順位: stanza_package が渡されていればそれを採用、そうでなければ package
    if stanza_package is not None:
        package = stanza_package

    # 従来互換：la で package 未指定なら perseus
    if lang == "la" and package is None:
        package = "perseus"

    import stanza  # lazy import

    pipe_kwargs: dict[str, Any] = {
        "lang": lang,
        "processors": processors,
        "package": package,
        "use_gpu": use_gpu,
    }
    if stanza_dir:
        pipe_kwargs["dir"] = stanza_dir
    if download_method is not None:
        pipe_kwargs["download_method"] = download_method

    # 呼び出し側から stanza.Pipeline に追加で渡したいものがあれば許容
    # （例: tokenize_pretokenized=True など）
    pipe_kwargs.update(kwargs)

    try:
        return stanza.Pipeline(**pipe_kwargs)
    except Exception:
        if not auto_download:
            # 従来実装はここで download→再試行していたが、
            # auto_download=False の場合は例外をそのまま返す（デグレではなく挙動選択）
            raise

        # 従来互換：失敗したら download して再試行
        # package が dict の場合 stanza.download の引数として扱いにくいので、
        # download は lang 単位で呼び出す（stanza が必要なものを解決する）
        try:
            stanza.download(lang, package=package)  # type: ignore[arg-type]
        except TypeError:
            # Mapping 指定などで stanza.download が受けない場合のフォールバック
            stanza.download(lang)

        return stanza.Pipeline(**pipe_kwargs)


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
    *,
    ref_tag_detector: Optional[Callable[[str], str]] = None,
    ref_tag_counter: Optional[Counter] = None,
) -> Counter:
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
    ref_tag_detector: Optional[Callable[[str], str]] = None,
    ref_tag_counter: Optional[Counter] = None,
) -> Counter:
    """
    Trace path: dump evidence rows to TSV while counting nouns.

    TSV columns (MUST match tests/test_trace_offsets.py):
      label
      chunk
      sent_idx
      token_idx
      token_char_start_in_chunk
      token_char_start_in_text
      sentence
      token
      lemma
      upos
      ref_tag
      global_row

    - token_char_start_in_text = chunk_base_offset + token_char_start_in_chunk
    - If trace_max_rows > 0 and the limit is reached:
        → stop writing trace rows
        → continue counting normally
    """
    total = Counter()
    if not text:
        return total

    trace_tsv.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    trace_enabled = True
    global_row = 0

    chunk_base_offset = 0  # cumulative offset in the full text

    header = [
        "label",
        "chunk",
        "sent_idx",
        "token_idx",
        "token_char_start_in_chunk",
        "token_char_start_in_text",
        "sentence",
        "token",
        "lemma",
        "upos",
        "ref_tag",
        "global_row",
    ]

    with trace_tsv.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp, delimiter="\t")
        w.writerow(header)

        for k, chunk in enumerate(iter_char_chunks(text, chunk_chars=chunk_chars), 1):
            doc = nlp(chunk)

            for s_i, sent in enumerate(getattr(doc, "sentences", []) or [], 1):
                sent_text = getattr(sent, "text", "") if trace_enabled else ""

                for t_i, (wd, start_in_chunk) in enumerate(_iter_sentence_words_with_offsets(sent), 1):
                    upos = getattr(wd, "upos", None)
                    if upos not in upos_targets:
                        continue

                    token = getattr(wd, "text", None)
                    lemma = getattr(wd, "lemma", None)

                    key = lemma if (use_lemma and lemma) else token
                    key = (key or "").strip().lower()

                    tag = ""
                    if key and ref_tag_detector is not None:
                        tag = ref_tag_detector(key)
                        if tag:
                            if ref_tag_counter is not None:
                                ref_tag_counter[tag] += 1
                        else:
                            total[key] += 1
                    elif key:
                        total[key] += 1

                    if trace_enabled:
                        # offset columns
                        sic = "" if start_in_chunk is None else str(start_in_chunk)
                        sit = (
                            ""
                            if start_in_chunk is None
                            else str(chunk_base_offset + int(start_in_chunk))
                        )

                        global_row += 1
                        w.writerow(
                            [
                                label or "",
                                k,
                                s_i,
                                t_i,
                                sic,
                                sit,
                                sent_text,
                                token or "",
                                lemma or "",
                                upos or "",
                                tag,
                                global_row,
                            ]
                        )
                        rows_written += 1
                        if trace_max_rows and rows_written >= trace_max_rows:
                            trace_enabled = False

            if label:
                print(f"[NLP] {label}: chunk {k} processed (chars {len(chunk):,})")

            # advance base offset after processing this chunk
            chunk_base_offset += len(chunk)

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