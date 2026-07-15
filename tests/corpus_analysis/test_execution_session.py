from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from nlpo_toolkit.corpus_analysis.analysis_policy import AnalysisExtractionPolicy
from nlpo_toolkit.corpus_analysis.config import load_config
from nlpo_toolkit.corpus_analysis.execution_session import (
    CorpusExecutionSession,
    NLPExecutionSession,
    prepare_analysis_corpus_session,
    start_nlp_execution_session,
)
from nlpo_toolkit.corpus_analysis.ports import (
    CorpusExecutionDependencies,
    CorpusPlanningDependencies,
    CorpusPreparationDependencies,
    NLPExecutionDependencies,
)
from nlpo_toolkit.corpus_analysis.requests import CorpusPreparationRequest
from nlpo_toolkit.latin.cleaners.config_loader import inspect_cleaner_config
from tests.corpus_analysis.fake_nlp import fake_backend_factory


def _corpus_dependencies() -> CorpusExecutionDependencies:
    return CorpusExecutionDependencies(
        planning=CorpusPlanningDependencies(load_config, inspect_cleaner_config),
        preparation=CorpusPreparationDependencies(
            lambda: (_ for _ in ()).throw(AssertionError("cleaner must not run"))
        ),
    )


def test_corpus_session_has_only_canonical_plan_and_corpora(tmp_path: Path) -> None:
    (tmp_path / "input.txt").write_text("Rosa amat", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text("groups:\n  text: {files: [input.txt]}\n", encoding="utf-8")
    session = prepare_analysis_corpus_session(
        CorpusPreparationRequest(tmp_path, config_path),
        dependencies=_corpus_dependencies(),
    )
    assert {field.name for field in fields(CorpusExecutionSession)} == {"plan", "corpora"}
    assert tuple(corpus.label for corpus in session.corpora) == ("text",)
    assert session.corpora[0].prepared_text == "Rosa amat"


def test_nlp_session_keeps_built_backend_whole(tmp_path: Path) -> None:
    (tmp_path / "input.txt").write_text("Rosa", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text("groups:\n  text: {files: [input.txt]}\n", encoding="utf-8")
    corpus = prepare_analysis_corpus_session(
        CorpusPreparationRequest(tmp_path, config_path),
        dependencies=_corpus_dependencies(),
    )
    calls = []
    factory = fake_backend_factory()

    def recording_factory(config):
        calls.append(config)
        return factory(config)

    policy = AnalysisExtractionPolicy(chunk_chars=123)
    session = start_nlp_execution_session(
        corpus,
        dependencies=NLPExecutionDependencies(recording_factory, policy),
    )
    assert {field.name for field in fields(NLPExecutionSession)} == {
        "corpus", "backend", "extraction_policy", "roman_exceptions"
    }
    assert session.corpus is corpus
    assert session.extraction_policy is policy
    assert session.roman_exceptions == frozenset()
    assert len(calls) == 1
