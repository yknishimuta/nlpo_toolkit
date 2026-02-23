from __future__ import annotations

import re
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from typing import Pattern as TypingPattern

import yaml

BASE_DIR = Path(__file__).resolve().parent
CORPUS_CORPORUM_YAML_PATH = BASE_DIR.parent / "cleaners" / "patterns" / "corpus_corporum.yml"
SCHOLASTIC_TEXT_YAML_PATH = BASE_DIR.parent / "cleaners" / "patterns" / "scholastic_text.yml"

# End of initial metadata section: a line consisting of 5 or more '#'
HEADER_HASH_RE = re.compile(r"^\s*#{5,}\s*$")

Pattern = TypingPattern[str]


@dataclass(frozen=True)
class LineRemoveRule:
    """
    Remove an entire line if pattern matches.
    Extra metadata (ref/name) are preserved for tracing / auditing.
    """
    pattern: Pattern
    ref: Any = ""
    name: str = ""


@dataclass(frozen=True)
class SubstituteRule:
    """
    Substitute occurrences in a line: pattern.sub(repl, line)
    Extra metadata (ref/name) are preserved for tracing / auditing.
    """
    pattern: Pattern
    repl: str
    ref: Any = ""
    name: str = ""


@dataclass(frozen=True)
class RefEvent:
    """
    A single rule-hit event for later analysis (standard schema).
    """
    doc_id: str
    kind: str               # "corpus_corporum" or "scholastic_text"
    rule_name: str
    action: str             # "drop_line" or "substitute"
    line_no: int            # 1-based line number (relative to processed region)
    match_count: int

    ref_key: str
    ref_author: str
    ref_work: str
    ref_loc: str

    text_snippet: str


def flatten_ref(ref: Any) -> dict[str, str]:
    """
    Accept ref as either:
      - string: "Aristotle:Metaphys"
      - dict: {key, author, work, loc}
    Return a flat dict for TSV columns:
      ref_key, ref_author, ref_work, ref_loc
    """
    out = {"ref_key": "", "ref_author": "", "ref_work": "", "ref_loc": ""}

    if ref is None:
        return out

    if isinstance(ref, str):
        s = ref.strip()
        out["ref_key"] = s
        # Optional light parsing: "Author:Work"
        if ":" in s:
            a, w = s.split(":", 1)
            out["ref_author"] = a.strip()
            out["ref_work"] = w.strip()
        return out

    if isinstance(ref, dict):
        out["ref_key"] = str(ref.get("key", "") or "").strip()
        out["ref_author"] = str(ref.get("author", "") or "").strip()
        out["ref_work"] = str(ref.get("work", "") or "").strip()
        out["ref_loc"] = str(ref.get("loc", "") or "").strip()

        # If key is missing, synthesize from author/work
        if not out["ref_key"] and (out["ref_author"] or out["ref_work"]):
            out["ref_key"] = f'{out["ref_author"]}:{out["ref_work"]}'.strip(":")
        return out

    # Fallback
    out["ref_key"] = str(ref).strip()
    return out


def load_patterns_from_yaml(yaml_path: str | Path) -> dict:
    """
    Common loader that reads remove_line_patterns / substitute_patterns from YAML
    and compiles them as re.Pattern objects.

    Supported keys per item:
      - enabled: bool (default True)
      - pattern: str  (required)
      - repl: str     (substitute_patterns only; default "")
      - ref: str|dict (optional; metadata, also used for TSV logging)
      - name: str     (optional; metadata)
    """
    yaml_path = Path(yaml_path)
    data: Dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    remove_line_rules: List[LineRemoveRule] = []
    for item in data.get("remove_line_patterns", []):
        if item.get("enabled", True):
            pat = re.compile(item["pattern"])
            ref = item.get("ref", "")
            name = str(item.get("name", "") or "")
            remove_line_rules.append(LineRemoveRule(pattern=pat, ref=ref, name=name))

    substitute_rules: List[SubstituteRule] = []
    for item in data.get("substitute_patterns", []):
        if item.get("enabled", True):
            pat = re.compile(item["pattern"])
            repl = item.get("repl", "")
            ref = item.get("ref", "")
            name = str(item.get("name", "") or "")
            substitute_rules.append(SubstituteRule(pattern=pat, repl=repl, ref=ref, name=name))

    return {
        "remove_line_patterns": remove_line_rules,
        "substitute_patterns": substitute_rules,
    }


def _write_ref_events_tsv(ref_tsv: Path, events: List[RefEvent]) -> None:
    """
    Append events to TSV (create with header if missing).
    """
    ref_tsv.parent.mkdir(parents=True, exist_ok=True)
    exists = ref_tsv.exists()

    with ref_tsv.open("a", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp, delimiter="\t")
        if not exists:
            w.writerow(
                [
                    "doc_id",
                    "kind",
                    "rule_name",
                    "action",
                    "line_no",
                    "match_count",
                    "ref_key",
                    "ref_author",
                    "ref_work",
                    "ref_loc",
                    "text_snippet",
                ]
            )
        for e in events:
            w.writerow(
                [
                    e.doc_id,
                    e.kind,
                    e.rule_name,
                    e.action,
                    e.line_no,
                    e.match_count,
                    e.ref_key,
                    e.ref_author,
                    e.ref_work,
                    e.ref_loc,
                    e.text_snippet,
                ]
            )

def clean_text(
    text: str,
    *,
    kind: str,
    ref_tsv=None,
    doc_id: str = "",
    rules_path=None,
) -> str:
    kind = (kind or "").strip()

    if kind == "corpus_corporum":
        return clean_corpus_corporum_text(
            text,
            yaml_path=rules_path,
            ref_tsv=ref_tsv,
            doc_id=doc_id,
        )

    if kind == "scholastic_text":
        return clean_scholastic_text(
            text,
            yaml_path=rules_path,
            ref_tsv=ref_tsv,
            doc_id=doc_id,
        )

    raise ValueError(f"Unknown cleaner kind: {kind!r}")

def clean_corpus_corporum_text(
    text: str,
    yaml_path: str | Path = CORPUS_CORPORUM_YAML_PATH,
    *,
    ref_tsv: Optional[str | Path] = None,
    doc_id: str = "",
    snippet_chars: int = 200,
) -> str:
    if yaml_path is None:
        yaml_path = CORPUS_CORPORUM_YAML_PATH

    patterns = load_patterns_from_yaml(yaml_path)
    remove_line_rules: List[LineRemoveRule] = patterns["remove_line_patterns"]
    substitute_rules: List[SubstituteRule] = patterns["substitute_patterns"]

    lines = text.splitlines()

    # Skip the initial metadata block (until the "######" line)
    start_idx = 0
    for i, line in enumerate(lines):
        if HEADER_HASH_RE.match(line):
            start_idx = i + 1
            break

    cleaned_lines: list[str] = []
    events: List[RefEvent] = []

    # Process each line in the main body
    for rel_i, raw_line in enumerate(lines[start_idx:], 1):
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        # Remove the entire line if it matches any delete-pattern
        dropped = False
        for rule in remove_line_rules:
            if rule.pattern.match(stripped):
                dropped = True
                if ref_tsv is not None:
                    flat = flatten_ref(rule.ref)
                    events.append(
                        RefEvent(
                            doc_id=doc_id,
                            kind="corpus_corporum",
                            rule_name=rule.name,
                            action="drop_line",
                            line_no=rel_i,
                            match_count=1,
                            ref_key=flat["ref_key"],
                            ref_author=flat["ref_author"],
                            ref_work=flat["ref_work"],
                            ref_loc=flat["ref_loc"],
                            text_snippet=line[:snippet_chars],
                        )
                    )
                break
        if dropped:
            continue

        # Apply substitution patterns to the line
        for rule in substitute_rules:
            # count matches BEFORE substitution (so we can log only when it actually hit)
            m = rule.pattern.findall(line)
            if m:
                line = rule.pattern.sub(rule.repl, line)
                if ref_tsv is not None:
                    flat = flatten_ref(rule.ref)
                    events.append(
                        RefEvent(
                            doc_id=doc_id,
                            kind="corpus_corporum",
                            rule_name=rule.name,
                            action="substitute",
                            line_no=rel_i,
                            match_count=len(m),
                            ref_key=flat["ref_key"],
                            ref_author=flat["ref_author"],
                            ref_work=flat["ref_work"],
                            ref_loc=flat["ref_loc"],
                            text_snippet=stripped[:snippet_chars],
                        )
                    )

        line = line.replace("\t", " ")
        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    cleaned_text = re.sub(r" {2,}", " ", cleaned_text)

    if ref_tsv is not None:
        _write_ref_events_tsv(Path(ref_tsv), events)

    return cleaned_text.strip() + "\n"


def clean_scholastic_text(
    text: str,
    yaml_path: str | Path = SCHOLASTIC_TEXT_YAML_PATH,
    *,
    ref_tsv: Optional[str | Path] = None,
    doc_id: str = "",
    snippet_chars: int = 200,
) -> str:
    if yaml_path is None:
        yaml_path = CORPUS_CORPORUM_YAML_PATH

    patterns = load_patterns_from_yaml(yaml_path)
    remove_line_rules: List[LineRemoveRule] = patterns["remove_line_patterns"]
    substitute_rules: List[SubstituteRule] = patterns["substitute_patterns"]

    lines = text.splitlines()
    cleaned_lines: list[str] = []
    events: List[RefEvent] = []

    for rel_i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Remove the line if any deletion pattern matches
        dropped = False
        for rule in remove_line_rules:
            if rule.pattern.match(stripped):
                dropped = True
                if ref_tsv is not None:
                    flat = flatten_ref(rule.ref)
                    events.append(
                        RefEvent(
                            doc_id=doc_id,
                            kind="scholastic_text",
                            rule_name=rule.name,
                            action="drop_line",
                            line_no=rel_i,
                            match_count=1,
                            ref_key=flat["ref_key"],
                            ref_author=flat["ref_author"],
                            ref_work=flat["ref_work"],
                            ref_loc=flat["ref_loc"],
                            text_snippet=line[:snippet_chars],
                        )
                    )
                break
        if dropped:
            continue

        # Apply substitution patterns
        for rule in substitute_rules:
            m = rule.pattern.findall(line)
            if m:
                line = rule.pattern.sub(rule.repl, line)
                if ref_tsv is not None:
                    flat = flatten_ref(rule.ref)
                    events.append(
                        RefEvent(
                            doc_id=doc_id,
                            kind="scholastic_text",
                            rule_name=rule.name,
                            action="substitute",
                            line_no=rel_i,
                            match_count=len(m),
                            ref_key=flat["ref_key"],
                            ref_author=flat["ref_author"],
                            ref_work=flat["ref_work"],
                            ref_loc=flat["ref_loc"],
                            text_snippet=stripped[:snippet_chars],
                        )
                    )

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    cleaned_text = re.sub(r" {2,}", " ", cleaned_text)

    if ref_tsv is not None:
        _write_ref_events_tsv(Path(ref_tsv), events)

    return cleaned_text.strip() + "\n"