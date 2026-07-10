from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.cache_storage import (
    CacheLockTimeout,
    acquire_cache_lock,
    release_cache_lock,
)


def test_acquire_and_release_cache_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "locks" / "aa" / "object.lock"

    acquire_cache_lock(lock_path, timeout_sec=0.2)

    assert lock_path.exists()
    assert "pid=" in lock_path.read_text(encoding="utf-8")

    release_cache_lock(lock_path)

    assert not lock_path.exists()


def test_release_missing_cache_lock_is_noop(tmp_path: Path) -> None:
    release_cache_lock(tmp_path / "missing.lock")


def test_cache_lock_times_out_when_lock_exists(tmp_path: Path) -> None:
    lock_path = tmp_path / "object.lock"
    lock_path.write_text("pid=99999\n", encoding="utf-8")

    with pytest.raises(CacheLockTimeout):
        acquire_cache_lock(
            lock_path,
            timeout_sec=0.05,
            poll_sec=0.01,
        )


def test_cache_modules_share_lock_and_prune_types() -> None:
    from nlpo_toolkit.corpus_analysis import analysis_cache
    from nlpo_toolkit.corpus_analysis import cache_storage
    from nlpo_toolkit.corpus_analysis import lemma_cache

    assert analysis_cache.CacheLockTimeout is cache_storage.CacheLockTimeout
    assert lemma_cache.CacheLockTimeout is cache_storage.CacheLockTimeout

    assert analysis_cache.PruneReport is cache_storage.PruneReport
    assert lemma_cache.PruneReport is cache_storage.PruneReport


def test_analysis_cache_does_not_import_lemma_cache() -> None:
    path = Path("nlpo_toolkit/corpus_analysis/analysis_cache.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.level == 1 and node.module == "lemma_cache")
            assert node.module != "nlpo_toolkit.corpus_analysis.lemma_cache"
        if isinstance(node, ast.Import):
            assert all(
                alias.name != "nlpo_toolkit.corpus_analysis.lemma_cache"
                for alias in node.names
            )


def test_cache_storage_has_no_payload_module_dependencies() -> None:
    path = Path("nlpo_toolkit/corpus_analysis/cache_storage.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {
        "analysis_cache",
        "config",
        "lemma_cache",
        "runner",
        "token_artifact",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module.split(".")[-1] not in forbidden
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[-1] not in forbidden


def test_importing_analysis_cache_does_not_load_lemma_cache() -> None:
    code = """
import sys
import nlpo_toolkit.corpus_analysis.analysis_cache
assert "nlpo_toolkit.corpus_analysis.lemma_cache" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_legacy_lemma_cache_reexports_shared_types() -> None:
    from nlpo_toolkit.corpus_analysis import cache_storage
    from nlpo_toolkit.corpus_analysis import lemma_cache
    from nlpo_toolkit.count_vocabula import lemma_cache as old_lemma_cache

    assert lemma_cache.CacheLockTimeout is cache_storage.CacheLockTimeout
    assert old_lemma_cache.CacheLockTimeout is cache_storage.CacheLockTimeout

    assert lemma_cache.PruneReport is cache_storage.PruneReport
    assert old_lemma_cache.PruneReport is cache_storage.PruneReport
