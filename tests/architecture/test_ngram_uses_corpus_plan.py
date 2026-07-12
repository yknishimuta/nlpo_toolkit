from __future__ import annotations

import ast
from pathlib import Path


NGRAM_PATH = Path("nlpo_toolkit/corpus_analysis/ngram.py")


def test_ngram_uses_canonical_corpus_planning_boundary() -> None:
    tree = ast.parse(NGRAM_PATH.read_text(encoding="utf-8"), filename=str(NGRAM_PATH))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert {"load_config", "run_preprocess_if_needed", "resolve_corpus_work_items"}.isdisjoint(imported)
    assert {"build_corpus_plan", "prepare_corpora"} <= imported


def test_ngram_has_no_cleaner_fallback_or_direct_cleaner_execution() -> None:
    paths = (
        NGRAM_PATH,
        Path("nlpo_toolkit/corpus_analysis/cli/ngram.py"),
    )
    forbidden = ("SimpleNamespace", "main=lambda argv: 0", "clean_mod")
    offenders = [
        (str(path), fragment)
        for path in paths
        for fragment in forbidden
        if fragment in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
