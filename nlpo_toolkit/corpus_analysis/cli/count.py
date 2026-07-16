from __future__ import annotations

import argparse
from pathlib import Path

from ..count_command import CountCommandError, CountRequest, execute_count_command
from ..composition import default_count_command_dependencies
from ..dry_run import DiagnosticLevel, execute_dry_run
from .common import (
    CLIContext,
    add_empty_group_argument,
    add_grouping_override_arguments,
    add_project_config_arguments,
    build_corpus_preparation_request,
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
    for mismatch in result.run.partition_mismatches:
        print(
            f"[{mismatch.level}] partition {mismatch.name} mismatch: "
            f"token_delta={mismatch.token_delta} mismatched_items={mismatch.mismatched_items}",
            file=stderr,
        )
    archive = result.archive
    if archive is None:
        return
    try:
        archive_directory = archive.archive_directory.relative_to(project_root)
    except ValueError:
        archive_directory = archive.archive_directory
    print(f"[ARCHIVE] saved run archive: {archive_directory}", file=stdout)
    print(f"[ARCHIVE] included input files: {archive.copied_files.inputs}", file=stdout)
    print(f"[ARCHIVE] included cleaned files: {archive.copied_files.cleaned}", file=stdout)

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
    add_grouping_override_arguments(parser)
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
    add_empty_group_argument(parser)


def execute(args: argparse.Namespace, context: CLIContext) -> int:
    corpus_request = build_corpus_preparation_request(args)
    try:
        dependencies = default_count_command_dependencies()
        if args.dry_run:
            result = execute_dry_run(
                corpus_request,
                dependencies=dependencies.runner.planning,
            )
            _present_dry_run(result, stdout=context.stdout)
            return 0 if result.successful else 1
        request = CountRequest(
            corpus=corpus_request,
            command_line=tuple(context.argv),
            archive_run=bool(args.archive_run),
            run_name=args.run_name,
            runs_dir=args.runs_dir,
            include_cleaned=bool(args.include_cleaned),
            include_input=bool(args.include_input),
        )
        result = execute_count_command(request, dependencies=dependencies)
        _present_count_result(
            result,
            project_root=request.corpus.project_root,
            stdout=context.stdout,
            stderr=context.stderr,
        )
        return 0 if result.successful else 1
    except CountCommandError as exc:
        present_error(exc, stderr=context.stderr)
        return 1
