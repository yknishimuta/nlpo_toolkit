from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml

BASE_DIR = Path(__file__).resolve().parent
CORPUS_CORPORUM_YAML_PATH = BASE_DIR.parent / "cleaners" / "patterns" / "corpus_corporum.yml"
SCHOLASTIC_TEXT_YAML_PATH = BASE_DIR.parent / "cleaners" / "patterns" / "scholastic_text.yml"

# End of initial metadata section: a line consisting of 5 or more '#'
HEADER_HASH_RE = re.compile(r'^\s*#{5,}\s*$')

Pattern = re.Pattern[str]

def load_patterns_from_yaml(yaml_path: str | Path) -> dict:
    """
    Common loader that reads remove_line_patterns / substitute_patterns from YAML
    and compiles them as re.Pattern objects.
    """
    yaml_path = Path(yaml_path)
    data: Dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    remove_line_patterns: List[Pattern] = []
    for item in data.get("remove_line_patterns", []):
        if item.get("enabled", True):
            remove_line_patterns.append(re.compile(item["pattern"]))

    substitute_patterns: List[Tuple[Pattern, str]] = []
    for item in data.get("substitute_patterns", []):
        if item.get("enabled", True):
            pattern = re.compile(item["pattern"])
            repl = item.get("repl", "")
            substitute_patterns.append((pattern, repl))

    return {
        "remove_line_patterns": remove_line_patterns,
        "substitute_patterns": substitute_patterns,
    }

def clean_text(text: str, kind: str) -> str:
    if kind == "corpus_corporum":
        return clean_corpus_corporum_text(text)
    elif kind == "scholastic_text":
        return clean_scholastic_text(text)
    else:
        raise ValueError(f"Unknown cleaner kind: {kind}")

def clean_corpus_corporum_text(
    text: str,
    yaml_path: str | Path = CORPUS_CORPORUM_YAML_PATH,
) -> str:

    patterns = load_patterns_from_yaml(yaml_path)
    remove_line_patterns: List[re.Pattern[str]] = patterns["remove_line_patterns"]
    substitute_patterns: List[Tuple[re.Pattern[str], str]] = patterns["substitute_patterns"]

    lines = text.splitlines()

    # Skip the initial metadata block (until the "######" line)
    start_idx = 0
    for i, line in enumerate(lines):
        if HEADER_HASH_RE.match(line):
            start_idx = i + 1
            break

    cleaned_lines: list[str] = []

    # 2) Process each line in the main body
    for raw_line in lines[start_idx:]:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        # Remove the entire line if it matches any delete-pattern
        if any(pat.match(stripped) for pat in remove_line_patterns):
            continue

        #  Apply substitution patterns to the line
        for pat, repl in substitute_patterns:
            line = pat.sub(repl, line)


        line = line.replace('\t', ' ')

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)

    return cleaned_text.strip() + "\n"

def clean_scholastic_text(
    text: str,
    yaml_path: str | Path = SCHOLASTIC_TEXT_YAML_PATH,
) -> str:

    patterns = load_patterns_from_yaml(yaml_path)
    remove_line_patterns: List[re.Pattern[str]] = patterns["remove_line_patterns"]
    substitute_patterns: List[Tuple[re.Pattern[str], str]] = patterns["substitute_patterns"]

    lines = text.splitlines()
    cleaned_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Remove the line if any deletion pattern matches
        if any(pat.match(stripped) for pat in remove_line_patterns):
            continue

        # Apply substitution patterns
        for pat, repl in substitute_patterns:
            line = pat.sub(repl, line)

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)

    return cleaned_text.strip() + "\n"