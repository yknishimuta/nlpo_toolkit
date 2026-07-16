from __future__ import annotations

import glob
from collections.abc import Sequence
from pathlib import Path

from .corpus_errors import CorpusReadError


def ensure_out_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def expand_globs(patterns: Sequence[str]) -> list[Path]:
    files = []
    for pat in patterns:
        files.extend(Path(p) for p in glob.glob(pat, recursive=True))
    return sorted({p.resolve() for p in files if p.is_file()})


def read_concat(paths: Sequence[Path]) -> str:
    chunks: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise CorpusReadError(path, str(exc)) from exc
        chunks.append(text)
    return "\n".join(chunks)
