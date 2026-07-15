from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from re import Pattern
from typing import Literal

from nlpo_toolkit.cleaner_contracts import CleanerKind


RuleAction = Literal["drop_line", "substitute"]


@dataclass(frozen=True)
class RuleReference:
    key: str = ""
    author: str = ""
    work: str = ""
    location: str = ""


@dataclass(frozen=True)
class LineRemoveRule:
    pattern: Pattern[str]
    reference: RuleReference = RuleReference()
    name: str = ""


@dataclass(frozen=True)
class SubstituteRule:
    pattern: Pattern[str]
    replacement: str
    reference: RuleReference = RuleReference()
    name: str = ""


@dataclass(frozen=True)
class RuleSet:
    remove_lines: tuple[LineRemoveRule, ...] = ()
    substitutions: tuple[SubstituteRule, ...] = ()


@dataclass(frozen=True)
class RefEvent:
    doc_id: str
    kind: CleanerKind
    rule_name: str
    action: RuleAction
    line_number: int
    match_count: int
    reference: RuleReference
    text_snippet: str


@dataclass(frozen=True)
class RuleApplicationResult:
    lines: tuple[str, ...]
    events: tuple[RefEvent, ...]


@dataclass(frozen=True)
class CleaningResult:
    text: str
    events: tuple[RefEvent, ...]


@dataclass(frozen=True)
class CleanerProfile:
    kind: CleanerKind
    default_rules_path: Path
    prepare_lines: Callable[[str], tuple[str, ...]]
    finalize_line: Callable[[str], str]


@dataclass(frozen=True)
class CleanerProgram:
    profile: CleanerProfile
    rules: RuleSet
    lexicon_map: Mapping[str, str]
    snippet_chars: int = 200
