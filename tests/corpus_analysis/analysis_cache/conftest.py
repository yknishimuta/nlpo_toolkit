from __future__ import annotations

from nlpo_toolkit.corpus_analysis.analysis_cache.models import AnalysisFingerprint
from nlpo_toolkit.corpus_analysis.analysis_records import NLPAnalysisRecord


def fingerprint(**changes: object) -> AnalysisFingerprint:
    values = {
        "backend": "fake", "language": "la", "processors": ("tokenize", "pos"),
        "chunk_size": 100, "chunk_strategy": "fixed", "package": "perseus",
    }
    values.update(changes)
    return AnalysisFingerprint(**values)  # type: ignore[arg-type]


def record(token: str = "Rosa") -> NLPAnalysisRecord:
    return NLPAnalysisRecord(
        chunk_index=0, sentence_index=0, token_index=0, global_token_index=0,
        char_start_in_chunk=None, char_end_in_chunk=None,
        char_start_in_text=None, char_end_in_text=None,
        sentence="Rosa amat", token=token, lemma=None, upos=None,
    )
