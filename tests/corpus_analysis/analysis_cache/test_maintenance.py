from __future__ import annotations

import os
import time
from pathlib import Path

from nlpo_toolkit.corpus_analysis.analysis_cache.maintenance import prune_analysis_cache


def test_prune_removes_old_pair_and_stale_lock(tmp_path: Path) -> None:
    shard = tmp_path / "objects/aa"
    shard.mkdir(parents=True)
    payload = shard / "a.jsonl"
    metadata = shard / "a.meta.json"
    payload.write_text("payload", encoding="utf-8")
    metadata.write_text("metadata", encoding="utf-8")
    lock = tmp_path / "locks/aa/a.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("lock", encoding="utf-8")
    old = time.time() - 10_000
    for path in (payload, metadata, lock):
        os.utime(path, (old, old))

    report = prune_analysis_cache(
        tmp_path, keep_days=0, keep_objects=0, lock_ttl_sec=1
    )

    assert not payload.exists() and not metadata.exists() and not lock.exists()
    assert report.removed_objects == 1
    assert report.removed_locks == 1
    assert report.bytes_freed == len("payloadmetadatalock")
