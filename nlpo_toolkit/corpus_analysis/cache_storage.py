"""Low-level cache storage primitives shared by cache implementations.

This module is independent of cache payload formats.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CacheLockTimeout",
    "PruneReport",
    "acquire_cache_lock",
    "release_cache_lock",
]


class CacheLockTimeout(RuntimeError):
    """Raised when a cache lock cannot be acquired before the timeout."""


@dataclass(frozen=True)
class PruneReport:
    """Summary of cache object, lock, and empty directory cleanup."""

    removed_objects: int
    removed_locks: int
    removed_empty_dirs: int
    bytes_freed: int


def acquire_cache_lock(
    lock_path: Path,
    *,
    timeout_sec: float = 300.0,
    poll_sec: float = 0.1,
) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    pid = os.getpid()

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"pid={pid}\n".encode("utf-8"))
            finally:
                os.close(fd)
            return
        except FileExistsError:
            if (time.time() - start) >= timeout_sec:
                raise CacheLockTimeout(f"Timeout acquiring lock: {lock_path}")
            time.sleep(poll_sec)


def release_cache_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass
