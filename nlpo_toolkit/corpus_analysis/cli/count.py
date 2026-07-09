from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from nlpo_toolkit.backends import create_nlp_backend

from ..archive import RunArchiveError, create_run_archive
from ..config import ensure_app_config, load_config
from ..dry_run import dry_run_count_vocabula
from ..nlp_hooks import (
    build_pipeline,
    build_sentence_splitter,
    count_group,
    render_stanza_package_table,
)
from ..runner import run
from .common import (
    CLIContext,
    add_project_config_arguments,
    resolve_config_path,
    resolve_project_root,
    set_handler,
)


_DEFAULT_BUILD_PIPELINE = build_pipeline
_DEFAULT_BUILD_SENTENCE_SPLITTER = build_sentence_splitter

try:
    from nlpo_toolkit.latin.cleaners import run_clean_corpus as clean_mod
except Exception:
    clean_mod = SimpleNamespace(main=lambda argv: 0)


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
) -> int:
    try:
        legacy_build_pipeline = build_pipeline if build_pipeline is not _DEFAULT_BUILD_PIPELINE else None
        legacy_sentence_splitter = (
            build_sentence_splitter
            if build_sentence_splitter is not _DEFAULT_BUILD_SENTENCE_SPLITTER
            else None
        )
        rc = run(
            project_root=project_root,
            config_path=config_path,
            group_by_file=group_by_file,
            load_config_fn=load_config,
            clean_mod=clean_mod,
            build_pipeline_fn=legacy_build_pipeline,
            backend_factory=None if legacy_build_pipeline is not None else create_nlp_backend,
            build_sentence_splitter_fn=legacy_sentence_splitter,
            count_group_fn=count_group,
            render_stanza_package_table_fn=render_stanza_package_table,
            error_on_empty_group=error_on_empty_group,
            auto_single_cleaned=auto_single_cleaned,
        )
    except ValueError as exc:
        if auto_single_cleaned and "--auto-single-cleaned" in str(exc):
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        raise

    cfg = ensure_app_config(load_config(config_path))
    archive_enabled = cfg.archive.enabled
    should_archive = archive_run or bool(run_name) or archive_enabled
    if rc != 0 or not should_archive:
        return rc

    effective_runs_dir = runs_dir if runs_dir is not None else Path(cfg.archive.runs_dir)
    effective_include_input = include_input or cfg.archive.include_input
    effective_include_cleaned = include_cleaned or cfg.archive.include_cleaned

    try:
        run_dir = create_run_archive(
            project_root=project_root,
            config_path=config_path,
            config=cfg,
            run_name=run_name,
            runs_dir=effective_runs_dir,
            include_cleaned=effective_include_cleaned,
            include_input=effective_include_input,
            command_line=command_line,
        )
    except (RunArchiveError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    try:
        display_run_dir = run_dir.relative_to(project_root)
    except ValueError:
        display_run_dir = run_dir

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    print(f"[ARCHIVE] saved run archive: {display_run_dir}")
    print(f"[ARCHIVE] included input files: {len(manifest.get('copied_input_files', []))}")
    print(f"[ARCHIVE] included cleaned files: {len(manifest.get('copied_cleaned_files', []))}")
    return rc


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
    )
