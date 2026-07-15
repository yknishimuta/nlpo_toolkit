from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from nlpo_toolkit.cleaner_contracts import CleanerKind

from .lexicon import EMPTY_LEXICON_MAP, apply_lexicon_map, load_lexicon_map
from .models import CleanerProgram, CleanerProfile, CleaningResult, RuleSet
from .normalization import normalize_cleaned_text
from .registry import get_cleaner_profile
from .rule_engine import apply_rule_set
from .rule_loader import load_rule_set


def clean_document(text: str, *, profile: CleanerProfile, rules: RuleSet, lexicon_map: Mapping[str, str], doc_id: str = "", snippet_chars: int = 200) -> CleaningResult:
    applied = apply_rule_set(profile.prepare_lines(text), rules=rules, kind=profile.kind, doc_id=doc_id, snippet_chars=snippet_chars, finalize_line=profile.finalize_line)
    cleaned = apply_lexicon_map(normalize_cleaned_text(applied.lines), lexicon_map)
    return CleaningResult(cleaned, applied.events)


def load_cleaner_program(*, kind: CleanerKind, rules_path: Path | None, lexicon_map_path: Path | None) -> CleanerProgram:
    profile = get_cleaner_profile(kind)
    rules = load_rule_set(rules_path or profile.default_rules_path)
    lexicon = load_lexicon_map(lexicon_map_path) if lexicon_map_path is not None else EMPTY_LEXICON_MAP
    return CleanerProgram(profile, rules, lexicon)
