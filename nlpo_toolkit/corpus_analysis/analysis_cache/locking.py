from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..cache_storage import acquire_cache_lock, release_cache_lock


@contextmanager
def held_cache_lock(lock_path: Path, *, timeout_sec: float) -> Iterator[None]:
    acquire_cache_lock(lock_path, timeout_sec=timeout_sec)
    try:
        yield
    finally:
        release_cache_lock(lock_path)
