from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

from nlpo_toolkit.corpus_analysis.features import FeatureOptions


def test_default_chunk_size_has_one_source() -> None:
    allowed = Path("nlpo_toolkit/corpus_analysis/analysis_policy.py")
    offenders: list[tuple[str, int]] = []
    for path in Path("nlpo_toolkit").rglob("*.py"):
        if path == allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == 200_000:
                offenders.append((str(path), node.lineno))
    assert offenders == []


def test_cache_has_no_independent_chunk_defaults() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("nlpo_toolkit/corpus_analysis/analysis_cache").glob("*.py")
    )
    assert "DEFAULT_CHUNK_SIZE" not in source
    assert "DEFAULT_CHUNK_STRATEGY" not in source
    assert "DEFAULT_PROCESSORS" not in source


def test_feature_options_uses_extraction_policy() -> None:
    names = {field.name for field in fields(FeatureOptions)}
    assert "chunk_chars" not in names
    assert "extraction_policy" in names
