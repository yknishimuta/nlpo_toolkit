from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .archive.errors import RunArchiveError
from .archive_types import RunArchiveRequest, RunArchiveResult
from .config_references import ConfigReferenceError
from .corpus_errors import CorpusPreparationError
from .ports import CountCommandDependencies
from .runner_types import RunResult
from .requests import CorpusPreparationRequest


class CountCommandError(RuntimeError):
    pass


@dataclass(frozen=True)
class CountRequest:
    corpus: CorpusPreparationRequest
    command_line: tuple[str, ...] = ()
    archive_run: bool = False
    run_name: str | None = None
    runs_dir: Path | None = None
    include_input: bool = False
    include_cleaned: bool = False


@dataclass(frozen=True)
class CountCommandResult:
    run: RunResult
    archive: RunArchiveResult | None

    @property
    def successful(self) -> bool:
        return self.run.exit_code == 0


def execute_count_command(
    request: CountRequest,
    *,
    dependencies: CountCommandDependencies,
) -> CountCommandResult:
    try:
        result = dependencies.run_analysis(
            request.corpus,
            dependencies=dependencies.runner,
        )
    except (
        ConfigReferenceError,
        CorpusPreparationError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        raise CountCommandError(str(exc)) from exc

    config = result.plan.definition.config
    should_archive = (
        request.archive_run
        or bool(request.run_name)
        or config.archive.enabled
    )
    if result.exit_code != 0 or not should_archive:
        return CountCommandResult(run=result, archive=None)

    archive_root = (
        request.runs_dir
        if request.runs_dir is not None
        else Path(config.archive.runs_dir)
    )
    try:
        archive_result = dependencies.archive_creator(
            run_result=result,
            request=RunArchiveRequest(
                archive_root=archive_root,
                run_name=request.run_name,
                include_cleaned_files=(
                    request.include_cleaned
                    or config.archive.include_cleaned
                ),
                include_input_files=(
                    request.include_input
                    or config.archive.include_input
                ),
                command_line=request.command_line,
            ),
        )
    except (RunArchiveError, ValueError) as exc:
        raise CountCommandError(str(exc)) from exc

    return CountCommandResult(run=result, archive=archive_result)
