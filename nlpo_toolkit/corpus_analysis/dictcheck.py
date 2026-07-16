from __future__ import annotations
from pathlib import Path

def load_lemma_normalize_map(path: Path) -> dict[str, str]:
    m: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split("\t")
        if len(parts) != 2:
            raise ValueError(f"lemma normalize TSV must have 2 columns: {path} line={line!r}")
        src, dst = parts[0].strip(), parts[1].strip()
        if src and dst:
            m[src] = dst
    return m
