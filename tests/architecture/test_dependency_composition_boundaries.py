from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, MISSING, fields, is_dataclass
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis import composition, ports
from nlpo_toolkit.corpus_analysis.archive import create_run_archive
from nlpo_toolkit.corpus_analysis.config import NLPConfig, load_config
from nlpo_toolkit.corpus_analysis.runner import run


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PORTS_PATH = PROJECT_ROOT / "nlpo_toolkit/corpus_analysis/ports.py"
COMPOSITION_PATH = PROJECT_ROOT / "nlpo_toolkit/corpus_analysis/composition.py"
CONTAINERS = (
    ports.CorpusPlanningDependencies,
    ports.AnalysisDependencies,
    ports.RunnerDependencies,
    ports.CountCommandDependencies,
    ports.FeatureCommandDependencies,
    ports.ConfigNgramDependencies,
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _function_parameters(function) -> tuple[tuple[str, inspect._ParameterKind], ...]:
    return tuple(
        (name, parameter.kind)
        for name, parameter in inspect.signature(function).parameters.items()
        if name != "self"
    )


def test_ports_contains_only_interfaces_and_dependency_containers() -> None:
    tree = _tree(PORTS_PATH)
    function_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert function_names == set()
    assert not any(name.startswith("default_") for name in function_names)
    assert imported_names.isdisjoint(
        {
            "create_nlp_backend",
            "build_sentence_splitter",
            "load_config",
            "load_default_cleaner",
            "create_run_archive",
            "run",
        }
    )
    assert _imported_modules(PORTS_PATH).isdisjoint(
        {
            "composition",
            "runner",
            "archive",
            "cleaner_runtime",
        }
    )


def test_composition_owns_production_factories_but_no_ports() -> None:
    tree = _tree(COMPOSITION_PATH)
    default_functions = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("default_")
    }
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]

    assert default_functions == {
        "default_corpus_planning_dependencies",
        "default_analysis_dependencies",
        "default_runner_dependencies",
        "default_count_command_dependencies",
        "default_feature_command_dependencies",
        "default_config_ngram_dependencies",
    }
    assert classes == []
    assert "ports" in _imported_modules(COMPOSITION_PATH)


def test_dependency_containers_are_frozen_and_have_no_production_defaults() -> None:
    for container in CONTAINERS:
        assert is_dataclass(container)
        assert container.__dataclass_params__.frozen is True

    planning_fields = {field.name: field for field in fields(ports.CorpusPlanningDependencies)}
    assert set(planning_fields) == {"load_config", "cleaner_loader", "cleaner_inspector"}
    assert all(
        field.default is MISSING and field.default_factory is MISSING
        for field in planning_fields.values()
    )

    planning = ports.CorpusPlanningDependencies(
        load_config=lambda _path: None,  # type: ignore[return-value]
        cleaner_loader=lambda: None,  # type: ignore[return-value]
        cleaner_inspector=lambda _path: None,  # type: ignore[return-value]
    )
    with pytest.raises(FrozenInstanceError):
        planning.load_config = planning.load_config


def test_runner_and_archive_ports_match_concrete_signatures() -> None:
    assert _function_parameters(ports.CountRunner.__call__) == _function_parameters(run)
    assert _function_parameters(ports.ArchiveCreator.__call__) == _function_parameters(
        create_run_archive
    )


def test_application_services_use_ports_not_composition() -> None:
    service_paths = (
        "runner.py",
        "runtime.py",
        "run_plan.py",
        "count_command.py",
        "features.py",
        "ngram.py",
        "dry_run.py",
    )
    for filename in service_paths:
        path = PROJECT_ROOT / "nlpo_toolkit/corpus_analysis" / filename
        imports = _imported_modules(path)
        assert "ports" in imports, filename
        assert "composition" not in imports, filename


def test_only_cli_bootstrap_uses_production_composition() -> None:
    cli_paths = (
        PROJECT_ROOT / "nlpo_toolkit/corpus_analysis/cli/count.py",
        PROJECT_ROOT / "nlpo_toolkit/corpus_analysis/cli/features.py",
        PROJECT_ROOT / "nlpo_toolkit/corpus_analysis/cli/ngram.py",
    )
    assert all("composition" in _imported_modules(path) for path in cli_paths)

    offenders: list[str] = []
    for path in (PROJECT_ROOT / "nlpo_toolkit/corpus_analysis").glob("*.py"):
        if path.name == "composition.py":
            continue
        if "composition" in _imported_modules(path):
            offenders.append(path.name)
    assert offenders == []


def test_test_helpers_build_ports_without_production_composition() -> None:
    helper = PROJECT_ROOT / "tests/corpus_analysis/fake_nlp.py"
    imports = _imported_modules(helper)
    assert "nlpo_toolkit.corpus_analysis.ports" in imports
    assert "nlpo_toolkit.corpus_analysis.composition" not in imports


def test_legacy_dependencies_module_and_imports_are_absent() -> None:
    assert not (PROJECT_ROOT / "nlpo_toolkit/corpus_analysis/dependencies.py").exists()
    offenders: list[str] = []
    for path in PROJECT_ROOT.rglob("*.py"):
        if "corpus_analysis.dependencies" in _imported_modules(path):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []


def test_production_composition_selects_expected_implementations() -> None:
    planning = composition.default_corpus_planning_dependencies()
    count = composition.default_count_command_dependencies()

    assert planning.load_config is load_config
    assert planning.cleaner_loader is composition.load_default_cleaner
    assert planning.cleaner_inspector is composition._inspect_cleaner_config
    assert count.run_analysis is run
    assert count.archive_creator is create_run_archive
    assert set(field.name for field in fields(type(count.runner))) == {"planning", "analysis"}
    assert set(field.name for field in fields(ports.FeatureCommandDependencies)) == {
        "planning",
        "analysis",
    }
    assert set(field.name for field in fields(ports.ConfigNgramDependencies)) == {"planning"}


def test_analysis_composition_binds_the_same_extraction_policy(monkeypatch) -> None:
    policy = composition.AnalysisExtractionPolicy(chunk_chars=1234)
    received = {}
    sentinel = object()

    def fake_create_backend(config, *, extraction_policy):
        received.update(config=config, extraction_policy=extraction_policy)
        return sentinel

    monkeypatch.setattr(composition, "create_nlp_backend", fake_create_backend)
    dependencies = composition.default_analysis_dependencies(extraction_policy=policy)
    config = NLPConfig()

    assert dependencies.backend_factory(config) is sentinel
    assert dependencies.extraction_policy is policy
    assert received == {"config": config, "extraction_policy": policy}


def test_sentence_splitter_adapter_passes_nlp_configuration(monkeypatch) -> None:
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        composition,
        "StanzaBackend",
        lambda **kwargs: calls.append(kwargs) or sentinel,
    )
    config = NLPConfig(language="grc", stanza_package=None, cpu_only=True)

    assert composition._create_sentence_splitter(config) is sentinel
    assert calls == [
        {
            "lang": "grc",
            "package": "perseus",
            "use_gpu": False,
            "processors": "tokenize",
        }
    ]


def test_default_factories_do_not_return_global_container_singletons() -> None:
    first = composition.default_runner_dependencies()
    second = composition.default_runner_dependencies()
    assert first is not second
    assert first.planning is not second.planning
    assert first.analysis is not second.analysis


def test_fresh_interpreter_imports_all_dependency_consumers() -> None:
    modules = (
        "nlpo_toolkit.corpus_analysis.ports",
        "nlpo_toolkit.corpus_analysis.composition",
        "nlpo_toolkit.corpus_analysis.runner",
        "nlpo_toolkit.corpus_analysis.runtime",
        "nlpo_toolkit.corpus_analysis.count_command",
        "nlpo_toolkit.corpus_analysis.features",
        "nlpo_toolkit.corpus_analysis.ngram",
    )
    code = "; ".join(f"import {module}" for module in modules)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_ports_and_composition_preserve_bundled_cleaner_lazy_loading() -> None:
    code = """
import sys
import nlpo_toolkit.corpus_analysis.ports
assert 'nlpo_toolkit.latin.cleaners.run_clean_corpus' not in sys.modules
import nlpo_toolkit.corpus_analysis.composition as composition
assert 'nlpo_toolkit.latin.cleaners.run_clean_corpus' not in sys.modules
composition.load_default_cleaner()
assert 'nlpo_toolkit.latin.cleaners.run_clean_corpus' in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
