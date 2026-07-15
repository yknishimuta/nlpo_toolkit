from pathlib import Path

from ..models import CleanerProfile


DEFAULT_RULES_PATH = Path(__file__).resolve().parents[1] / "patterns" / "scholastic_text.yml"


def prepare_lines(text: str) -> tuple[str, ...]:
    return tuple(text.splitlines())


def finalize_line(line: str) -> str:
    return line


PROFILE = CleanerProfile("scholastic_text", DEFAULT_RULES_PATH, prepare_lines, finalize_line)
