from __future__ import annotations

import time
from pathlib import Path

from ..cache_storage import PruneReport
from .keys import cache_metadata_path


def prune_analysis_cache(
    cache_dir: Path,
    *,
    keep_days: int = 30,
    keep_objects: int = 50_000,
    lock_ttl_sec: int = 3600,
) -> PruneReport:
    cache_dir = Path(cache_dir).resolve()
    objects_dir = cache_dir / "objects"
    locks_dir = cache_dir / "locks"
    now = time.time()
    cutoff_ts = now - (keep_days * 86400)
    removed_objects = removed_locks = removed_empty_dirs = bytes_freed = 0

    if locks_dir.exists():
        for path in locks_dir.rglob("*.lock"):
            try:
                stat = path.stat()
                if (now - stat.st_mtime) > lock_ttl_sec:
                    bytes_freed += stat.st_size
                    path.unlink(missing_ok=True)
                    removed_locks += 1
            except OSError:
                pass

    payloads = sorted(
        (path for path in objects_dir.rglob("*.jsonl") if path.is_file())
        if objects_dir.exists() else (),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    survivors = set(payloads[:max(0, int(keep_objects))])
    for payload in payloads:
        try:
            stat = payload.stat()
        except OSError:
            continue
        if payload in survivors or stat.st_mtime >= cutoff_ts:
            continue
        for path in (payload, cache_metadata_path(payload)):
            try:
                bytes_freed += path.stat().st_size
                path.unlink(missing_ok=True)
            except OSError:
                pass
        removed_objects += 1

    for base in (objects_dir, locks_dir):
        if not base.exists():
            continue
        directories = sorted(
            (path for path in base.rglob("*") if path.is_dir()),
            key=lambda path: len(str(path)), reverse=True,
        )
        for directory in directories:
            try:
                if not any(directory.iterdir()):
                    directory.rmdir()
                    removed_empty_dirs += 1
            except OSError:
                pass

    return PruneReport(removed_objects, removed_locks, removed_empty_dirs, bytes_freed)
