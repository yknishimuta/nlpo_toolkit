from pathlib import Path

import pytest

from nlpo_toolkit.cleaner_contracts import (
    CleanerApplicationError,
    CleanerExecutionResult,
)
from nlpo_toolkit.corpus_analysis.corpus_errors import CleanerExecutionError
from nlpo_toolkit.corpus_analysis.ports import CorpusPreparationDependencies
from nlpo_toolkit.corpus_analysis.preprocessing import CleanerPlan, execute_preprocess
from nlpo_toolkit.latin.cleaners.config_loader import inspect_cleaner_config


def _plan(tmp_path: Path) -> CleanerPlan:
    config = tmp_path / "cleaner.yml"
    source = tmp_path / "input.txt"
    source.write_text("input", encoding="utf-8")
    config.write_text(
        "kind: scholastic_text\ninput: input.txt\noutput: cleaned\n",
        encoding="utf-8",
    )
    return CleanerPlan(config, inspect_cleaner_config(config))


def _result(plan: CleanerPlan) -> CleanerExecutionResult:
    config = plan.inspection.config
    return CleanerExecutionResult(
        config.source_path, config.kind, config.output_path, (), config.ref_tsv_path
    )


def test_no_preprocess_does_not_call_cleaner_service() -> None:
    dependencies = CorpusPreparationDependencies(
        execute_cleaner=lambda _request: pytest.fail("cleaner must not run")
    )
    assert execute_preprocess(None, dependencies=dependencies) is None


def test_preprocess_passes_the_planned_inspection_and_returns_output(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    calls = []

    def service(request):
        calls.append(request)
        return _result(plan)

    output = execute_preprocess(
        plan, dependencies=CorpusPreparationDependencies(execute_cleaner=service)
    )
    assert calls[0].inspection is plan.inspection
    assert output == plan.inspection.config.output_path


def test_cleaner_application_error_is_wrapped_and_keeps_cause(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    failure = CleanerApplicationError("clean failed")

    def service(_request):
        raise failure

    with pytest.raises(CleanerExecutionError, match="clean failed") as caught:
        execute_preprocess(
            plan, dependencies=CorpusPreparationDependencies(execute_cleaner=service)
        )
    assert caught.value.__cause__ is failure


def test_unexpected_service_output_is_rejected(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    config = plan.inspection.config

    def service(_request):
        return CleanerExecutionResult(
            config.source_path, config.kind, tmp_path / "wrong", (), None
        )

    with pytest.raises(CleanerExecutionError, match="unexpected output path"):
        execute_preprocess(
            plan, dependencies=CorpusPreparationDependencies(execute_cleaner=service)
        )
