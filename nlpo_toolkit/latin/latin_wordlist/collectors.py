from __future__ import annotations

from collections import Counter
from pathlib import Path
import stat

from .engine import iter_latin_word_candidates
from .errors import LatinWordlistSourceReadError
from .models import (
    ConlluCandidates,
    ExtraWordlistCandidates,
    TextCandidates,
    WordlistNotice,
    WordlistNoticeCode,
    WordlistTokenizationPolicy,
)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise LatinWordlistSourceReadError(
            f"Failed to read Latin wordlist source {path} as UTF-8: {exc}"
        ) from exc


def _source_kind(path: Path, *, expected: str) -> bool:
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise LatinWordlistSourceReadError(
            f"Failed to inspect Latin wordlist source {path}: {exc}"
        ) from exc
    matches = stat.S_ISDIR(mode) if expected == "directory" else stat.S_ISREG(mode)
    if not matches:
        raise LatinWordlistSourceReadError(
            f"Latin wordlist source {path} is not a {expected}"
        )
    return True


def _sorted_source_files(directory: Path, pattern: str) -> tuple[Path, ...]:
    try:
        return tuple(sorted(directory.rglob(pattern)))
    except OSError as exc:
        raise LatinWordlistSourceReadError(
            f"Failed to enumerate Latin wordlist source directory {directory}: {exc}"
        ) from exc


def collect_conllu_candidates(
    *, directory: Path, min_length: int
) -> tuple[ConlluCandidates, tuple[WordlistNotice, ...]]:
    if not _source_kind(directory, expected="directory"):
        notice = WordlistNotice(
            WordlistNoticeCode.MISSING_CONLLU_DIRECTORY,
            directory,
            f"CoNLL-U directory not found: {directory}",
        )
        return ConlluCandidates((), frozenset(), {}), (notice,)

    files = _sorted_source_files(directory, "*.conllu")
    lemmas: set[str] = set()
    form_counts: Counter[str] = Counter()
    ignored_rows = 0
    for path in files:
        text = _read_text(path)
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            columns = stripped.split("\t")
            if len(columns) < 3:
                ignored_rows += 1
                continue
            form = columns[1].strip().lower()
            lemma = columns[2].strip().lower()
            if len(form) >= min_length and form.isalpha():
                form_counts[form] += 1
            if len(lemma) >= min_length and lemma.isalpha():
                lemmas.add(lemma)
    return ConlluCandidates(files, frozenset(lemmas), form_counts, ignored_rows), ()


def collect_text_candidates(
    *,
    directory: Path,
    policy: WordlistTokenizationPolicy,
    min_length: int,
) -> tuple[TextCandidates, tuple[WordlistNotice, ...]]:
    if not _source_kind(directory, expected="directory"):
        notice = WordlistNotice(
            WordlistNoticeCode.MISSING_TEXT_DIRECTORY,
            directory,
            f"Latin text directory not found: {directory}",
        )
        return TextCandidates((), {}), (notice,)

    files = _sorted_source_files(directory, "*.txt")
    form_counts: Counter[str] = Counter()
    for path in files:
        form_counts.update(
            iter_latin_word_candidates(
                _read_text(path), policy=policy, min_length=min_length
            )
        )
    return TextCandidates(files, form_counts), ()


def collect_extra_wordlist_candidates(
    *, path: Path
) -> tuple[ExtraWordlistCandidates | None, tuple[WordlistNotice, ...]]:
    if not _source_kind(path, expected="file"):
        notice = WordlistNotice(
            WordlistNoticeCode.MISSING_EXTRA_WORDLIST,
            path,
            f"Extra wordlist not found: {path}",
        )
        return None, (notice,)

    entries: set[str] = set()
    for line in _read_text(path).splitlines():
        candidate = line.strip()
        if not candidate or candidate.startswith("#"):
            continue
        word = candidate.lower()
        if word.isalpha():
            entries.add(word)
    return ExtraWordlistCandidates(path, frozenset(entries)), ()
