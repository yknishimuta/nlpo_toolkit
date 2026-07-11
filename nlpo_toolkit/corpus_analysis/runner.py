from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Tuple

from nlpo_toolkit.backends import BuiltNLPBackend

from . import analysis_pipeline, post_analysis, run_reporting, runtime
from .config import AppConfig
from .runner_types import RunnerDependencies


def run(
    *,
    project_root: Path | None = None,
    script_dir: Path | None = None,
    config_path: Path,
    group_by_file: Optional[bool] = None,
    load_config_fn: Callable[[Path], AppConfig | Mapping[str, object]],
    clean_mod: Any,
    build_pipeline_fn: Callable[[str, str, bool], Tuple[Any, str]] | None = None,
    backend_factory: Callable[[Any], BuiltNLPBackend] | None = None,
    build_sentence_splitter_fn: Optional[Callable[..., Any]] = None,
    render_stanza_package_table_fn: Callable[..., List[str]] | None = None,
    error_on_empty_group: bool = False,
    auto_single_cleaned: bool = False,
) -> int:
    """Core runner. Dependencies are injectable for CLI and tests."""
    dependencies = RunnerDependencies(
        load_config=load_config_fn,
        clean_module=clean_mod,
        build_pipeline=build_pipeline_fn,
        backend_factory=backend_factory,
        build_sentence_splitter=build_sentence_splitter_fn,
        render_stanza_package_table=render_stanza_package_table_fn
        or (lambda *_args, **_kwargs: []),
    )

    context = runtime.prepare_run_context(
        project_root=project_root,
        script_dir=script_dir,
        config_path=config_path,
        group_by_file=group_by_file,
        auto_single_cleaned=auto_single_cleaned,
        error_on_empty_group=error_on_empty_group,
        dependencies=dependencies,
    )
    analysis = analysis_pipeline.analyze_corpora(context, dependencies)
    partitions = post_analysis.execute_partition_validations(
        context=context,
        analysis=analysis,
    )
    comparisons = post_analysis.execute_group_comparisons(
        context=context,
        analysis=analysis,
    )
    run_reporting.write_run_report(
        context=context,
        analysis=analysis,
        partitions=partitions,
        comparisons=comparisons,
        dependencies=dependencies,
    )
    return partitions.exit_code
