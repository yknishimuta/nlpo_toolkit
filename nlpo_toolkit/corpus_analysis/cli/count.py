from __future__ import annotations

import argparse
from pathlib import Path

from ..count_command import CountCommandError, CountRequest, execute_count_command
from ..dependencies import default_count_command_dependencies
from ..dry_run import DiagnosticLevel, execute_dry_run
from .common import (
    CLIContext,
    add_project_config_arguments,
    resolve_config_path,
    resolve_project_root,
    set_handler,
)
from .output import present_error


def _present_dry_run(result, *, stdout) -> None:
    prefixes = {
        DiagnosticLevel.OK: "[OK]",
        DiagnosticLevel.WARNING: "[WARN]",
        DiagnosticLevel.ERROR: "[ERROR]",
    }
    for diagnostic in result.diagnostics:
        print(f"{prefixes[diagnostic.level]} {diagnostic.message}", file=stdout)


def _present_count_result(result, *, project_root: Path, stdout, stderr) -> None:
    for name, level, token_delta, mismatched_items in result.run.partition_mismatches:
        print(
            f"[{level}] partition {name} mismatch: "
            f"token_delta={token_delta} mismatched_items={mismatched_items}",
            file=stderr,
        )
    archive = result.archive
    if archive is None:
        return
    try:
        run_dir = archive.run_dir.relative_to(project_root)
    except ValueError:
        run_dir = archive.run_dir
    print(f"[ARCHIVE] saved run archive: {run_dir}", file=stdout)
    print(f"[ARCHIVE] included input files: {archive.copied_input_count}", file=stdout)
    print(f"[ARCHIVE] included cleaned files: {archive.copied_cleaned_count}", file=stdout)

def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "count",
        help="Count corpus vocabulary and write frequency outputs.",
    )
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
        help="Archive this successful count run under --runs-dir.",
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

    request = CountRequest(
        project_root=project_root,
        config_path=config_path,
        command_line=tuple(context.argv),
        group_by_file=bool(args.group_by_file),
        archive_run=bool(args.archive_run),
        run_name=args.run_name,
        runs_dir=args.runs_dir,
        include_cleaned=bool(args.include_cleaned),
        include_input=bool(args.include_input),
        error_on_empty_group=bool(args.error_on_empty_group),
        auto_single_cleaned=bool(args.auto_single_cleaned),
        dry_run=bool(args.dry_run),
    )
    try:
        dependencies = default_count_command_dependencies()
        if request.dry_run:
            result = execute_dry_run(
                request=request,
                dependencies=dependencies.runner.planning,
            )
            _present_dry_run(result, stdout=context.stdout)
            return 0 if result.successful else 1
        result = execute_count_command(request, dependencies=dependencies)
        _present_count_result(
            result,
            project_root=request.project_root,
            stdout=context.stdout,
            stderr=context.stderr,
        )
        return 0 if result.successful else 1
    except CountCommandError as exc:
        present_error(exc, stderr=context.stderr)
        return 1
