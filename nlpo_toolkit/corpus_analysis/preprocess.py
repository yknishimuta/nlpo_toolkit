from __future__ import annotations

from pathlib import Path


def expand_cleaned_dir_placeholders(
    patterns: list[str],
    cleaned_dir: Path | None,
) -> list[str]:
    if cleaned_dir is None:
        return patterns
    return [pattern.replace("{cleaned_dir}", str(cleaned_dir)) for pattern in patterns]
