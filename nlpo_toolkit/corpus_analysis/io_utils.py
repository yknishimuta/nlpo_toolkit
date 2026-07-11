from __future__ import annotations
from pathlib import Path
from typing import List
import glob, sys

def expand_globs(patterns: List[str]) -> List[Path]:
    files = []
    for pat in patterns:
        files.extend(Path(p) for p in glob.glob(pat, recursive=True))
    return sorted({p.resolve() for p in files if p.is_file()})

def read_concat(paths: List[Path]) -> str:
    chunks: List[str] = []
    for p in paths:
        try:
            chunks.append(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}", file=sys.stderr)
    return "\n".join(chunks)
