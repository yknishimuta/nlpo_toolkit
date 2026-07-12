from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .archive import ArchiveOptions, RunArchiveError
from .cleaner_runtime import CleanerError
from .corpus_errors import CorpusPreparationError
from .dependencies import CountCommandDependencies
from .dry_run import execute_dry_run


class CountCommandError(RuntimeError):
    pass


@dataclass(frozen=True)
class CountRequest:
    project_root: Path
    config_path: Path
    command_line: tuple[str, ...] = ()
    group_by_file: bool = False
    auto_single_cleaned: bool = False
    error_on_empty_group: bool = False
    archive_run: bool = False
    run_name: str | None = None
    runs_dir: Path | None = None
    include_input: bool = False
    include_cleaned: bool = False
    dry_run: bool = False


def execute_count_command(
    request: CountRequest,
    *,
    dependencies: CountCommandDependencies,
) -> int:
    if request.dry_run:
        return execute_dry_run(
            request=request,
            dependencies=dependencies.runner.planning,
        )

    try:
        result = dependencies.run_analysis(
            project_root=request.project_root,
            config_path=request.config_path,
            group_by_file=request.group_by_file,
            dependencies=dependencies.runner,
            error_on_empty_group=request.error_on_empty_group,
            auto_single_cleaned=request.auto_single_cleaned,
        )
    except (
        CleanerError,
        CorpusPreparationError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        raise CountCommandError(str(exc)) from exc

    config = result.plan.config
    should_archive = (
        request.archive_run
        or bool(request.run_name)
        or config.archive.enabled
    )
    if result.exit_code != 0 or not should_archive:
        return result.exit_code

    runs_dir = (
        request.runs_dir
        if request.runs_dir is not None
        else Path(config.archive.runs_dir)
    )
    try:
        archive_result = dependencies.archive_creator(
            result=result,
            options=ArchiveOptions(
                run_name=request.run_name,
                runs_dir=runs_dir,
                include_cleaned=(
                    request.include_cleaned
                    or config.archive.include_cleaned
                ),
                include_input=(
                    request.include_input
                    or config.archive.include_input
                ),
                command_line=request.command_line,
            ),
        )
    except (RunArchiveError, ValueError) as exc:
        raise CountCommandError(str(exc)) from exc

    try:
        display_run_dir = archive_result.run_dir.relative_to(
            request.project_root
        )
    except ValueError:
        display_run_dir = archive_result.run_dir
    print(f"[ARCHIVE] saved run archive: {display_run_dir}")
    print(
        "[ARCHIVE] included input files: "
        f"{archive_result.copied_input_count}"
    )
    print(
        "[ARCHIVE] included cleaned files: "
        f"{archive_result.copied_cleaned_count}"
    )
    return result.exit_code
