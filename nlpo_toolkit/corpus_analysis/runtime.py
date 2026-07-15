from __future__ import annotations

from nlpo_toolkit.nlp.contracts import BuiltNLPBackend, NLPBackend
from nlpo_toolkit.nlp.roman_numerals import load_roman_exceptions

from .config import AppConfig
from .corpus import prepare_corpora
from .io_utils import ensure_out_dir
from .ports import BackendFactory, RunnerDependencies
from .run_plan import ResolvedAnalysisPlan, build_count_plan, prepare_count_plan
from .requests import CorpusPreparationRequest
from .runner_types import RunContext


def build_nlp_runtime(
    *,
    config: AppConfig,
    backend_factory: BackendFactory,
) -> BuiltNLPBackend:
    return backend_factory(config.nlp)


def initialize_nlp_runtime(
    *,
    config: AppConfig,
    dependencies: RunnerDependencies,
) -> BuiltNLPBackend:
    return build_nlp_runtime(
        config=config,
        backend_factory=dependencies.analysis.backend_factory,
    )


def initialize_sentence_splitter(
    *,
    config: AppConfig,
    dependencies: RunnerDependencies,
) -> NLPBackend | None:
    if dependencies.analysis.sentence_splitter_factory is None:
        return None
    return dependencies.analysis.sentence_splitter_factory(config.nlp)


def load_roman_exceptions_for_run(
    *,
    plan: ResolvedAnalysisPlan,
) -> frozenset[str]:
    path = plan.config_files.path("filters.roman_exceptions_file")
    if path is None:
        return frozenset()
    return load_roman_exceptions(path)


def prepare_run_context(
    request: CorpusPreparationRequest,
    *,
    dependencies: RunnerDependencies,
) -> RunContext:
    definition = build_count_plan(
        request,
        dependencies=dependencies.planning,
    )
    plan = prepare_count_plan(definition, dependencies=dependencies.preparation)
    prepared_corpora = prepare_corpora(
        work_items=plan.work_items,
        config=plan.config,
        config_files=plan.config_files,
    )
    ensure_out_dir(plan.out_dir)
    roman_exceptions = load_roman_exceptions_for_run(plan=plan)
    analysis_backend = initialize_nlp_runtime(
        config=plan.config,
        dependencies=dependencies,
    )
    sentence_splitter = initialize_sentence_splitter(
        config=plan.config,
        dependencies=dependencies,
    )
    return RunContext(
        plan=plan,
        prepared_corpora=prepared_corpora,
        analysis_backend=analysis_backend,
        sentence_splitter=sentence_splitter,
        roman_exceptions=roman_exceptions,
        extraction_policy=dependencies.analysis.extraction_policy,
    )
