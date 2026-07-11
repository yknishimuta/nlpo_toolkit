from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nlpo_toolkit.backends import create_nlp_backend
from nlpo_toolkit.nlp import build_stanza_pipeline

from ..config import load_config
from ..cleaner_runtime import CleanerError
from ..features import FeatureError, run_features
from .common import (
    CLIContext,
    add_project_config_arguments,
    resolve_config_path,
    resolve_project_root,
    set_handler,
)


def build_pipeline(language: str, stanza_package: str, cpu_only: bool):
    backend = build_stanza_pipeline(
        lang=language,
        package=stanza_package,
        use_gpu=not cpu_only,
    )
    return backend, stanza_package


_DEFAULT_BUILD_PIPELINE = build_pipeline


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("features")
    add_project_config_arguments(
        parser,
        project_root_help="Project root used to resolve relative paths in the config.",
        config_help="YAML config path. Defaults to <project-root>/config/groups.config.yml.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV/TSV path. Defaults to standard output.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "tsv"),
        default="csv",
        help="Output format.",
    )
    parser.add_argument(
        "--field",
        choices=("lemma", "token"),
        default="lemma",
        help="Field used for MFW features.",
    )
    parser.add_argument(
        "--mfw",
        type=int,
        default=0,
        help="Add relative-frequency features for the top N most frequent words/lemmas.",
    )
    parser.add_argument(
        "--include-upos",
        dest="include_upos",
        action="store_true",
        default=True,
        help="Include UPOS count and ratio features. Enabled by default.",
    )
    parser.add_argument(
        "--no-upos",
        dest="include_upos",
        action="store_false",
        help="Disable UPOS features.",
    )
    parser.add_argument(
        "--include-basic",
        dest="include_basic",
        action="store_true",
        default=True,
        help="Include basic text statistics. Enabled by default.",
    )
    parser.add_argument(
        "--no-basic",
        dest="include_basic",
        action="store_false",
        help="Disable basic text statistics.",
    )
    parser.add_argument(
        "--group-by-file",
        action="store_true",
        help="Write one feature row per input file instead of one row per configured group.",
    )
    parser.add_argument(
        "--auto-single-cleaned",
        action="store_true",
        help="Use the only .txt file in cleaned_dir as the feature target; fail if zero or multiple files exist.",
    )
    parser.add_argument(
        "--error-on-empty-group",
        action="store_true",
        help="Fail when any configured group matches zero files.",
    )
    set_handler(parser, execute)


def execute(args: argparse.Namespace, context: CLIContext) -> int:
    project_root = resolve_project_root(args.project_root)
    config_path = resolve_config_path(project_root=project_root, config_path=args.config)
    try:
        legacy_build_pipeline = build_pipeline if build_pipeline is not _DEFAULT_BUILD_PIPELINE else None
        return run_features(
            project_root=project_root,
            config_path=config_path,
            out=args.out,
            output_format=args.format,
            field=args.field,
            mfw=args.mfw,
            include_upos=bool(args.include_upos),
            include_basic=bool(args.include_basic),
            group_by_file=bool(args.group_by_file),
            auto_single_cleaned=bool(args.auto_single_cleaned),
            error_on_empty_group=bool(args.error_on_empty_group),
            build_pipeline_fn=legacy_build_pipeline,
            backend_factory=None if legacy_build_pipeline is not None else create_nlp_backend,
            load_config_fn=load_config,
        )
    except (CleanerError, FeatureError, ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
