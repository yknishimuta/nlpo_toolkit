import re
from pathlib import Path

from ..models import CleanerProfile


HEADER_HASH_RE = re.compile(r"^\s*#{5,}\s*$")
DEFAULT_RULES_PATH = Path(__file__).resolve().parents[1] / "patterns" / "corpus_corporum.yml"


def prepare_lines(text: str) -> tuple[str, ...]:
    lines = text.splitlines()
    start = 0
    for index, line in enumerate(lines):
        if HEADER_HASH_RE.match(line):
            start = index + 1
            break
    return tuple(lines[start:])


def finalize_line(line: str) -> str:
    return line.replace("\t", " ")


PROFILE = CleanerProfile("corpus_corporum", DEFAULT_RULES_PATH, prepare_lines, finalize_line)
