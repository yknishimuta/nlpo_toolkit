from __future__ import annotations

from pathlib import Path
from typing import Any

from nlpo_toolkit.backends import NLPBackendInfo
from nlpo_toolkit.nlp import load_roman_exceptions

from .config import AppConfig
from .corpus import prepare_corpora
from .dependencies import BackendFactory, RunnerDependencies
from .run_plan import AnalysisPlan, build_count_plan, ensure_out_dir
from .runner_types import RunContext


def build_nlp_runtime(
    *,
    config: AppConfig,
    backend_factory: BackendFactory,
) -> tuple[Any, NLPBackendInfo, Any]:
    built_backend = backend_factory(config.nlp)
    return built_backend.backend, built_backend.info, built_backend.info.package


def initialize_nlp_runtime(
    *,
    config: AppConfig,
    dependencies: RunnerDependencies,
) -> tuple[Any, NLPBackendInfo, Any]:
    return build_nlp_runtime(
        config=config,
        backend_factory=dependencies.analysis.backend_factory,
    )


def initialize_sentence_splitter(
    *,
    config: AppConfig,
    dependencies: RunnerDependencies,
) -> Any | None:
    if dependencies.analysis.sentence_splitter_factory is None:
        return None
    return dependencies.analysis.sentence_splitter_factory(config.nlp)


def load_roman_exceptions_for_run(
    *,
    plan: AnalysisPlan,
) -> frozenset[str]:
    path = plan.config_files.path("filters.roman_exceptions_file")
    if path is None:
        return frozenset()
    return load_roman_exceptions(path)


def prepare_run_context(
    *,
    project_root: Path | None,
    script_dir: Path | None,
    config_path: Path,
    group_by_file: bool | None,
    auto_single_cleaned: bool,
    error_on_empty_group: bool,
    dependencies: RunnerDependencies,
) -> RunContext:
    plan = build_count_plan(
        project_root=project_root,
        script_dir=script_dir,
        config_path=config_path,
        group_by_file=group_by_file,
        auto_single_cleaned=auto_single_cleaned,
        error_on_empty_group=error_on_empty_group,
        dependencies=dependencies.planning,
        preprocess_mode="execute",
    )
    prepared_corpora = prepare_corpora(
        work_items=plan.work_items,
        config=plan.config,
        config_files=plan.config_files,
    )
    ensure_out_dir(plan.out_dir)
    roman_exceptions = load_roman_exceptions_for_run(plan=plan)
    nlp, backend_info, package = initialize_nlp_runtime(
        config=plan.config,
        dependencies=dependencies,
    )
    splitter_nlp = initialize_sentence_splitter(
        config=plan.config,
        dependencies=dependencies,
    )
    return RunContext(
        plan=plan,
        prepared_corpora=prepared_corpora,
        nlp=nlp,
        backend_info=backend_info,
        splitter_nlp=splitter_nlp,
        roman_exceptions=roman_exceptions,
        extraction_policy=dependencies.analysis.extraction_policy,
    )
