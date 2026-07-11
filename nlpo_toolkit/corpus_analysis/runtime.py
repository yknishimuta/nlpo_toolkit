from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from nlpo_toolkit.backends import BuiltNLPBackend, NLPBackendInfo, create_nlp_backend
from nlpo_toolkit.nlp import load_roman_exceptions

from .config import AppConfig
from .corpus import resolve_project_path, run_preprocess_if_needed
from .run_plan import build_run_plan, ensure_out_dir
from .runner_types import RunContext, RunnerDependencies


def build_nlp_runtime(
    *,
    config: AppConfig,
    backend_factory: Callable[[Any], BuiltNLPBackend] | None = None,
    build_pipeline_fn: Callable[[str, Any, bool], tuple[Any, Any]] | None = None,
) -> tuple[Any, NLPBackendInfo, Any]:
    language = config.nlp.language
    stanza_package = config.nlp.stanza_package
    cpu_only = config.nlp.cpu_only

    if backend_factory is not None:
        built_backend = backend_factory(config.nlp)
        return built_backend.backend, built_backend.info, built_backend.info.package

    if build_pipeline_fn is not None:
        nlp, package = build_pipeline_fn(language, stanza_package, cpu_only)
        return (
            nlp,
            NLPBackendInfo(
                name="stanza",
                language=language,
                package=package,
                use_gpu=not cpu_only,
            ),
            package,
        )

    built_backend = create_nlp_backend(config.nlp)
    return built_backend.backend, built_backend.info, built_backend.info.package


def initialize_nlp_runtime(
    *,
    config: AppConfig,
    dependencies: RunnerDependencies,
) -> tuple[Any, NLPBackendInfo, Any]:
    return build_nlp_runtime(
        config=config,
        backend_factory=dependencies.backend_factory,
        build_pipeline_fn=dependencies.build_pipeline,
    )


def initialize_sentence_splitter(
    *,
    config: AppConfig,
    package: Any,
    dependencies: RunnerDependencies,
) -> Any | None:
    if dependencies.build_sentence_splitter is None:
        return None
    try:
        return dependencies.build_sentence_splitter(
            config.nlp.language,
            stanza_package=package,
            cpu_only=config.nlp.cpu_only,
        )
    except Exception:
        return None


def load_roman_exceptions_for_run(
    *,
    config: AppConfig,
    project_root: Path,
) -> frozenset[str]:
    roman_exceptions_file = config.filters.roman_exceptions_file
    if not roman_exceptions_file:
        return frozenset()
    path = resolve_project_path(project_root, roman_exceptions_file)
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
    plan = build_run_plan(
        project_root=project_root,
        script_dir=script_dir,
        config_path=config_path,
        group_by_file=group_by_file,
        auto_single_cleaned=auto_single_cleaned,
        error_on_empty_group=error_on_empty_group,
        load_config_fn=dependencies.load_config,
        preprocess_mode="execute",
        clean_mod=dependencies.clean_module,
        preprocess_fn=run_preprocess_if_needed,
    )
    ensure_out_dir(plan.out_dir)
    nlp, backend_info, package = initialize_nlp_runtime(
        config=plan.config,
        dependencies=dependencies,
    )
    splitter_nlp = initialize_sentence_splitter(
        config=plan.config,
        package=package,
        dependencies=dependencies,
    )
    roman_exceptions = load_roman_exceptions_for_run(
        config=plan.config,
        project_root=plan.project_root,
    )
    return RunContext(
        plan=plan,
        nlp=nlp,
        backend_info=backend_info,
        stanza_package=package,
        splitter_nlp=splitter_nlp,
        roman_exceptions=roman_exceptions,
    )
