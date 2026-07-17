"""Execution session values shared by application orchestration and results."""

from __future__ import annotations

from dataclasses import dataclass

from nlpo_toolkit.nlp.contracts import BuiltNLPBackend

from .analysis_policy import AnalysisExtractionPolicy
from .corpus import PreparedCorpus
from .planning.models import ResolvedAnalysisPlan


@dataclass(frozen=True)
class CorpusExecutionSession:
    plan: ResolvedAnalysisPlan
    corpora: tuple[PreparedCorpus, ...]


@dataclass(frozen=True)
class NLPExecutionSession:
    corpus: CorpusExecutionSession
    backend: BuiltNLPBackend
    extraction_policy: AnalysisExtractionPolicy
    roman_exceptions: frozenset[str]

