from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..archive import ArchiveOptions, RunArchiveError, create_run_archive
from ..cleaner_runtime import CleanerError
from ..corpus_errors import CorpusPreparationError
from ..dependencies import default_runner_dependencies
from ..dry_run import dry_run_count_vocabula
from ..runner import run
from ..runner_types import RunnerDependencies
from .common import (
    CLIContext,
    add_project_config_arguments,
    resolve_config_path,
    resolve_project_root,
    set_handler,
)

def run_count_vocabula(
    *,
    project_root: Path,
    config_path: Path,
    group_by_file: bool = False,
    archive_run: bool = False,
    run_name: str | None = None,
    runs_dir: Path | None = None,
    include_cleaned: bool = False,
    include_input: bool = False,
    error_on_empty_group: bool = False,
    auto_single_cleaned: bool = False,
    command_line: list[str] | None = None,
    dependencies: RunnerDependencies,
) -> int:
    try:
        result = run(
            project_root=project_root,
            config_path=config_path,
            group_by_file=group_by_file,
            dependencies=dependencies,
            error_on_empty_group=error_on_empty_group,
            auto_single_cleaned=auto_single_cleaned,
        )
    except (CleanerError, CorpusPreparationError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        if auto_single_cleaned and "--auto-single-cleaned" in str(exc):
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        raise

    cfg = result.plan.config
    archive_enabled = cfg.archive.enabled
    should_archive = archive_run or bool(run_name) or archive_enabled
    if result.exit_code != 0 or not should_archive:
        return result.exit_code

    effective_runs_dir = runs_dir if runs_dir is not None else Path(cfg.archive.runs_dir)
    effective_include_input = include_input or cfg.archive.include_input
    effective_include_cleaned = include_cleaned or cfg.archive.include_cleaned

    try:
        archive_result = create_run_archive(
            result=result,
            options=ArchiveOptions(
                run_name=run_name,
                runs_dir=effective_runs_dir,
                include_cleaned=effective_include_cleaned,
                include_input=effective_include_input,
                command_line=tuple(command_line or ()),
            ),
        )
    except (RunArchiveError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    try:
        display_run_dir = archive_result.run_dir.relative_to(project_root)
    except ValueError:
        display_run_dir = archive_result.run_dir

    print(f"[ARCHIVE] saved run archive: {display_run_dir}")
    print(f"[ARCHIVE] included input files: {archive_result.copied_input_count}")
    print(f"[ARCHIVE] included cleaned files: {archive_result.copied_cleaned_count}")
    return result.exit_code


def register(subparsers: argparse._SubParsersAction) -> None:
    for name in ("count-vocabula", "count"):
        parser = subparsers.add_parser(name)
        _configure_parser(parser)
        set_handler(parser, execute)


def _configure_parser(parser: argparse.ArgumentParser) -> None:
    add_project_config_arguments(
        parser,
        project_root_help="Project root used to resolve relative paths in the config.",
        config_help="YAML config path. Defaults to <project-root>/config/groups.config.yml.",
    )
    parser.add_argument(
        "--group-by-file",
        action="store_true",
        help="Write one frequency CSV per input file instead of one CSV per configured group.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, paths, and matched files without running NLP.",
    )
    parser.add_argument(
        "--archive-run",
        action="store_true",
        help="Archive this successful count-vocabula run under --runs-dir.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run archive directory name. Implies --archive-run.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Directory where run archives are stored. Defaults to runs.",
    )
    parser.add_argument(
        "--include-cleaned",
        action="store_true",
        help="Copy cleaned files into the run archive.",
    )
    parser.add_argument(
        "--include-input",
        action="store_true",
        help="Copy input files into the run archive.",
    )
    parser.add_argument(
        "--error-on-empty-group",
        action="store_true",
        help="Fail when any configured group matches zero files.",
    )
    parser.add_argument(
        "--auto-single-cleaned",
        action="store_true",
        help="Use the only .txt file in cleaned_dir as the count target; fail if zero or multiple files exist.",
    )


def execute(args: argparse.Namespace, context: CLIContext) -> int:
    project_root = resolve_project_root(args.project_root)
    config_path = resolve_config_path(project_root=project_root, config_path=args.config)

    if args.dry_run:
        return dry_run_count_vocabula(
            project_root=project_root,
            config_path=config_path,
            group_by_file=bool(args.group_by_file),
            error_on_empty_group=bool(args.error_on_empty_group),
            auto_single_cleaned=bool(args.auto_single_cleaned),
        )

    return run_count_vocabula(
        project_root=project_root,
        config_path=config_path,
        group_by_file=bool(args.group_by_file),
        archive_run=bool(args.archive_run),
        run_name=args.run_name,
        runs_dir=args.runs_dir,
        include_cleaned=bool(args.include_cleaned),
        include_input=bool(args.include_input),
        error_on_empty_group=bool(args.error_on_empty_group),
        auto_single_cleaned=bool(args.auto_single_cleaned),
        command_line=list(context.argv),
        dependencies=default_runner_dependencies(),
    )
