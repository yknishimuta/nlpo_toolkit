from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.cleaner_runtime import (
    CleanerExecutionError,
    CleanerUnavailableError,
)
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
    return CleanerPlan(
        config_path=config,
        inspection=inspect_cleaner_config(config),
    )


def test_no_preprocess_does_not_load_cleaner() -> None:
    def failing_loader():
        raise AssertionError("cleaner must not be loaded")

    dependencies = CorpusPreparationDependencies(cleaner_loader=failing_loader)
    assert execute_preprocess(None, dependencies=dependencies) is None


def test_cleaner_loader_failure_is_not_silent(tmp_path: Path) -> None:
    def failing_loader():
        raise CleanerUnavailableError("cleaner unavailable")

    with pytest.raises(CleanerUnavailableError, match="cleaner unavailable"):
        execute_preprocess(
            _plan(tmp_path),
            dependencies=CorpusPreparationDependencies(cleaner_loader=failing_loader),
        )


def test_cleaner_exception_keeps_cause(tmp_path: Path) -> None:
    class FailingCleaner:
        @staticmethod
        def main(argv):
            raise RuntimeError("clean failed")

    with pytest.raises(CleanerExecutionError, match="clean failed") as caught:
        execute_preprocess(
            _plan(tmp_path),
            dependencies=CorpusPreparationDependencies(cleaner_loader=lambda: FailingCleaner()),
        )
    assert isinstance(caught.value.__cause__, RuntimeError)


@pytest.mark.parametrize("status", [1, 2, -1, True, False])
def test_failure_status_is_rejected(tmp_path: Path, status: object) -> None:
    class Cleaner:
        @staticmethod
        def main(argv):
            return status

    with pytest.raises(CleanerExecutionError, match="exit code"):
        execute_preprocess(
            _plan(tmp_path),
            dependencies=CorpusPreparationDependencies(cleaner_loader=lambda: Cleaner()),
        )


@pytest.mark.parametrize("status", [0, None])
def test_success_status_is_accepted(tmp_path: Path, status: object) -> None:
    class Cleaner:
        @staticmethod
        def main(argv):
            return status

    assert execute_preprocess(
        _plan(tmp_path),
        dependencies=CorpusPreparationDependencies(cleaner_loader=lambda: Cleaner()),
    ) == (
        tmp_path / "cleaned"
    ).resolve()


def test_missing_main_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CleanerUnavailableError, match="callable.*main"):
        execute_preprocess(
            _plan(tmp_path),
            dependencies=CorpusPreparationDependencies(cleaner_loader=object),
        )
