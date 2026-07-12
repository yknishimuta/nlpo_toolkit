from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.analysis_policy import (
    DEFAULT_ANALYSIS_EXTRACTION_POLICY,
)
from nlpo_toolkit.corpus_analysis.dependencies import (
    AnalysisDependencies,
    CorpusPlanningDependencies,
    RunnerDependencies,
)
from nlpo_toolkit.corpus_analysis.features import execute_feature_command
from nlpo_toolkit.corpus_analysis.ngram import write_ngrams_from_config
from nlpo_toolkit.corpus_analysis.runner import run


FORBIDDEN_PARAMETERS = {
    "load_config_fn",
    "clean_mod",
    "clean_module",
    "build_pipeline_fn",
    "build_sentence_splitter_fn",
    "count_group_fn",
    "render_stanza_package_table_fn",
}


def test_application_services_have_no_legacy_dependency_parameters() -> None:
    for function in (run, execute_feature_command, write_ngrams_from_config):
        parameters = set(inspect.signature(function).parameters)
        assert parameters.isdisjoint(FORBIDDEN_PARAMETERS)
        assert "dependencies" in parameters


def test_production_has_no_legacy_identity_injection() -> None:
    forbidden = (
        "_DEFAULT_",
        "legacy_build_pipeline",
        "legacy_sentence_splitter",
        " is not _DEFAULT",
    )
    offenders = []
    for path in Path("nlpo_toolkit").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden:
            if fragment in source:
                offenders.append((str(path), fragment))
    assert offenders == []


def test_cli_modules_do_not_import_low_level_dependencies() -> None:
    forbidden = {
        "load_config",
        "create_nlp_backend",
        "run_clean_corpus",
        "load_default_cleaner",
    }
    offenders = []
    for path in (
        Path("nlpo_toolkit/corpus_analysis/cli/count.py"),
        Path("nlpo_toolkit/corpus_analysis/cli/features.py"),
        Path("nlpo_toolkit/corpus_analysis/cli/ngram.py"),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in forbidden:
                        offenders.append((str(path), alias.name))
    assert offenders == []


def test_dependency_objects_are_frozen() -> None:
    planning = CorpusPlanningDependencies(
        load_config=lambda _path: None,  # type: ignore[return-value]
        cleaner_loader=lambda: None,  # type: ignore[return-value]
    )
    dependencies = RunnerDependencies(
        planning=planning,
        analysis=AnalysisDependencies(
            backend_factory=lambda _config: None,  # type: ignore[return-value]
            extraction_policy=DEFAULT_ANALYSIS_EXTRACTION_POLICY,
        ),
    )

    with pytest.raises(FrozenInstanceError):
        dependencies.analysis = dependencies.analysis
