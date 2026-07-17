from __future__ import annotations

from pathlib import Path


def load_lemma_normalization_map(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("\t")
        if len(parts) != 2:
            raise ValueError(
                f"lemma normalize TSV must have 2 columns: {path} line={line!r}"
            )
        source, destination = (part.strip() for part in parts)
        if source and destination:
            values[source] = destination
    return values
