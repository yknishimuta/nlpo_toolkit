from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

from nlpo_toolkit.corpus_analysis.execution_session import CorpusExecutionSession, NLPExecutionSession
from nlpo_toolkit.corpus_analysis.ports import ConfigNgramDependencies, FeatureCommandDependencies, RunnerDependencies
from nlpo_toolkit.corpus_analysis.runner_types import RunContext


ROOT = Path("nlpo_toolkit/corpus_analysis")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}


def test_session_and_dependency_fields_are_nonduplicating() -> None:
    assert {field.name for field in fields(CorpusExecutionSession)} == {"plan", "corpora"}
    assert {field.name for field in fields(NLPExecutionSession)} == {"corpus", "backend", "extraction_policy", "roman_exceptions"}
    assert {field.name for field in fields(RunContext)} == {"session", "sentence_splitter", "artifact_plan"}
    assert {field.name for field in fields(RunnerDependencies)} == {"corpus", "nlp", "count"}
    assert {field.name for field in fields(FeatureCommandDependencies)} == {"corpus", "nlp"}
    assert {field.name for field in fields(ConfigNgramDependencies)} == {"corpus"}


def test_commands_delegate_preparation_to_session_owner() -> None:
    forbidden = {
        ROOT / "features/service.py": {"build_analysis_plan", "prepare_analysis_plan", "prepare_corpora"},
        ROOT / "ngram.py": {"build_analysis_plan", "prepare_analysis_plan", "prepare_corpora", "start_nlp_execution_session"},
        ROOT / "runtime.py": {"build_count_plan", "prepare_count_plan", "prepare_corpora"},
    }
    for path, names in forbidden.items():
        assert _imports(path).isdisjoint(names)


def test_removed_runtime_helpers_are_absent() -> None:
    forbidden = {"build_nlp_runtime", "initialize_nlp_runtime", "load_roman_exceptions_for_run"}
    offenders = []
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders.extend((str(path), node.name) for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in forbidden)
    assert offenders == []
