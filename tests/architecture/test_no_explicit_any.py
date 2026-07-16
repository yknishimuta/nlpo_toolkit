from __future__ import annotations

import ast
from pathlib import Path
from typing import get_type_hints


ROOT = Path("nlpo_toolkit")


def test_production_has_no_explicit_any_annotations_or_imports():
    offenders = []
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "Any":
                offenders.append((path, node.lineno))
            if isinstance(node, ast.ImportFrom) and node.module == "typing":
                if any(alias.name == "Any" for alias in node.names):
                    offenders.append((path, node.lineno))
    assert offenders == []


def test_typed_models_own_previously_untyped_fields():
    from nlpo_toolkit.corpus_analysis.analysis_cache.models import (
        AnalysisCacheMetadata, AnalysisFingerprint,
    )
    from nlpo_toolkit.corpus_analysis.archive.models import ArchiveManifest
    from nlpo_toolkit.corpus_analysis.reporting.models import RunMetadata
    from nlpo_toolkit.corpus_analysis.token_artifact.schema import (
        TokenArtifactFilterDescriptor, TokenArtifactNLPDescriptor,
    )

    assert get_type_hints(AnalysisCacheMetadata)["fingerprint"] is AnalysisFingerprint
    assert "created_at" in ArchiveManifest.__annotations__
    assert "normalization" in RunMetadata.__annotations__
    assert TokenArtifactNLPDescriptor.model_fields
    assert TokenArtifactFilterDescriptor.model_fields
