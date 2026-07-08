from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

from nlpo_toolkit.compare import CompareError, run_compare

from .archive import RunArchiveError, create_run_archive
from .cache import CacheClearError, clear_cache
from .config import ensure_app_config, load_config
from .concordance import ConcordanceError, write_concordance
from .dry_run import dry_run_count_vocabula
from .features import FeatureError, run_features
from .ngram import NgramError, write_ngrams_from_config, write_ngrams_from_trace
from .nlp_hooks import (
    build_pipeline,
    build_sentence_splitter,
    count_group,
    render_stanza_package_table,
)
from .runner import run

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
        rc = run(
            project_root=project_root,
            config_path=config_path,
            group_by_file=group_by_file,
            load_config_fn=load_config,
            clean_mod=clean_mod,
            build_pipeline_fn=build_pipeline,
            build_sentence_splitter_fn=build_sentence_splitter,
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

    effective_runs_dir = runs_dir
    if effective_runs_dir is None:
        effective_runs_dir = Path(cfg.archive.runs_dir)

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
    import json

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    print(f"[ARCHIVE] saved run archive: {display_run_dir}")
    print(f"[ARCHIVE] included input files: {len(manifest.get('copied_input_files', []))}")
    print(f"[ARCHIVE] included cleaned files: {len(manifest.get('copied_cleaned_files', []))}")
    return rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nlpo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("count-vocabula", "count"):
        count_parser = subparsers.add_parser(name)
        count_parser.add_argument(
            "--project-root",
            type=Path,
            default=Path.cwd(),
            help="Project root used to resolve relative paths in the config.",
        )
        count_parser.add_argument(
            "--config",
            type=Path,
            default=None,
            help="YAML config path. Defaults to <project-root>/config/groups.config.yml.",
        )
        count_parser.add_argument(
            "--group-by-file",
            action="store_true",
            help="Write one frequency CSV per input file instead of one CSV per configured group.",
        )
        count_parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate config, paths, and matched files without running NLP.",
        )
        count_parser.add_argument(
            "--archive-run",
            action="store_true",
            help="Archive this successful count-vocabula run under --runs-dir.",
        )
        count_parser.add_argument(
            "--run-name",
            default=None,
            help="Run archive directory name. Implies --archive-run.",
        )
        count_parser.add_argument(
            "--runs-dir",
            type=Path,
            default=None,
            help="Directory where run archives are stored. Defaults to runs.",
        )
        count_parser.add_argument(
            "--include-cleaned",
            action="store_true",
            help="Copy cleaned files into the run archive.",
        )
        count_parser.add_argument(
            "--include-input",
            action="store_true",
            help="Copy input files into the run archive.",
        )
        count_parser.add_argument(
            "--error-on-empty-group",
            action="store_true",
            help="Fail when any configured group matches zero files.",
        )
        count_parser.add_argument(
            "--auto-single-cleaned",
            action="store_true",
            help="Use the only .txt file in cleaned_dir as the count target; fail if zero or multiple files exist.",
        )

    cache_parser = subparsers.add_parser("cache")
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command", required=True)
    cache_clear_parser = cache_subparsers.add_parser("clear")
    cache_clear_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root used to resolve the configured cache directory.",
    )
    cache_clear_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config path. Defaults to <project-root>/config/groups.config.yml when it exists.",
    )

    concordance_parser = subparsers.add_parser("concordance")
    concordance_parser.add_argument(
        "--trace",
        type=Path,
        required=True,
        help="Trace TSV path generated by count-vocabula.",
    )
    concordance_parser.add_argument(
        "--keys",
        nargs="+",
        required=True,
        help="Search keys. Multiple values are accepted.",
    )
    concordance_parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Number of words to show on each side of the matched token.",
    )
    concordance_parser.add_argument(
        "--field",
        choices=("token", "lemma"),
        default="lemma",
        help="Trace field to search.",
    )
    concordance_parser.add_argument(
        "--format",
        choices=("tsv", "csv"),
        default="tsv",
        help="Output format.",
    )
    concordance_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file path. Defaults to standard output.",
    )

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Two or more frequency CSV files to compare.",
    )
    compare_parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Display labels corresponding to --inputs.",
    )
    compare_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. Defaults to standard output.",
    )
    compare_parser.add_argument(
        "--format",
        choices=("csv", "tsv"),
        default="csv",
        help="Output format.",
    )
    compare_parser.add_argument(
        "--metric",
        choices=("relative", "difference", "ratio", "log-ratio"),
        default="log-ratio",
        help="Primary comparison metric. Rows include all available comparison columns.",
    )
    compare_parser.add_argument(
        "--smoothing",
        type=float,
        default=0.5,
        help="Additive smoothing used for ratio and log-ratio calculations.",
    )
    compare_parser.add_argument(
        "--sort",
        choices=("abs-log-ratio", "log-ratio", "difference", "range-relative", "total", "term"),
        default=None,
        help="Sort key. Defaults to abs-log-ratio for two inputs and range-relative for three or more.",
    )
    compare_parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending.",
    )
    compare_parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort descending. This is the default.",
    )
    compare_parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Limit output to the top N rows after sorting.",
    )
    compare_parser.add_argument(
        "--min-total-count",
        type=float,
        default=1,
        help="Exclude terms whose summed count across inputs is below this value.",
    )
    compare_parser.add_argument(
        "--key-column",
        default=None,
        help="Explicit key column name. Defaults to automatic detection.",
    )
    compare_parser.add_argument(
        "--count-column",
        default=None,
        help="Explicit count column name. Defaults to automatic detection.",
    )

    features_parser = subparsers.add_parser("features")
    features_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root used to resolve relative paths in the config.",
    )
    features_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config path. Defaults to <project-root>/config/groups.config.yml.",
    )
    features_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV/TSV path. Defaults to standard output.",
    )
    features_parser.add_argument(
        "--format",
        choices=("csv", "tsv"),
        default="csv",
        help="Output format.",
    )
    features_parser.add_argument(
        "--field",
        choices=("lemma", "token"),
        default="lemma",
        help="Field used for MFW features.",
    )
    features_parser.add_argument(
        "--mfw",
        type=int,
        default=0,
        help="Add relative-frequency features for the top N most frequent words/lemmas.",
    )
    features_parser.add_argument(
        "--include-upos",
        dest="include_upos",
        action="store_true",
        default=True,
        help="Include UPOS count and ratio features. Enabled by default.",
    )
    features_parser.add_argument(
        "--no-upos",
        dest="include_upos",
        action="store_false",
        help="Disable UPOS features.",
    )
    features_parser.add_argument(
        "--include-basic",
        dest="include_basic",
        action="store_true",
        default=True,
        help="Include basic text statistics. Enabled by default.",
    )
    features_parser.add_argument(
        "--no-basic",
        dest="include_basic",
        action="store_false",
        help="Disable basic text statistics.",
    )
    features_parser.add_argument(
        "--group-by-file",
        action="store_true",
        help="Write one feature row per input file instead of one row per configured group.",
    )
    features_parser.add_argument(
        "--auto-single-cleaned",
        action="store_true",
        help="Use the only .txt file in cleaned_dir as the feature target; fail if zero or multiple files exist.",
    )
    features_parser.add_argument(
        "--error-on-empty-group",
        action="store_true",
        help="Fail when any configured group matches zero files.",
    )

    ngram_parser = subparsers.add_parser("ngram")
    ngram_parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="N-gram size.",
    )
    ngram_parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="Trace TSV path generated by count-vocabula.",
    )
    ngram_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root used with --config input.",
    )
    ngram_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config path for groups input. Defaults to <project-root>/config/groups.config.yml.",
    )
    ngram_parser.add_argument(
        "--field",
        choices=("token", "lemma"),
        default="lemma",
        help="Trace field to use for n-grams.",
    )
    ngram_parser.add_argument(
        "--by-group",
        action="store_true",
        help="Aggregate n-grams separately for each trace group.",
    )
    ngram_parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum frequency to include.",
    )
    ngram_parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Limit output to the top N n-grams.",
    )
    ngram_parser.add_argument(
        "--format",
        choices=("tsv", "csv"),
        default="tsv",
        help="Output format.",
    )
    ngram_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file path. Defaults to standard output.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv_list)

    if args.command in {"count-vocabula", "count"}:
        project_root = args.project_root.resolve()
        config_path = args.config
        if config_path is None:
            config_path = project_root / "config" / "groups.config.yml"
        elif not config_path.is_absolute():
            config_path = (project_root / config_path).resolve()

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
            command_line=["nlpo", *argv_list],
        )

    if args.command == "cache" and args.cache_command == "clear":
        project_root = args.project_root.resolve()
        config_path = args.config
        if config_path is not None and not config_path.is_absolute():
            config_path = (project_root / config_path).resolve()
        try:
            return clear_cache(project_root=project_root, config_path=config_path)
        except CacheClearError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

    if args.command == "concordance":
        try:
            return write_concordance(
                trace_path=args.trace,
                keys=list(args.keys),
                field=args.field,
                window=args.window,
                output_format=args.format,
                out_path=args.out,
            )
        except ConcordanceError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

    if args.command == "compare":
        try:
            return run_compare(
                inputs=list(args.inputs),
                labels=list(args.labels) if args.labels is not None else None,
                out=args.out,
                output_format=args.format,
                metric=args.metric,
                smoothing=args.smoothing,
                min_total_count=args.min_total_count,
                top=args.top,
                sort=args.sort,
                ascending=bool(args.ascending) and not bool(args.descending),
                key_column=args.key_column,
                count_column=args.count_column,
            )
        except CompareError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

    if args.command == "features":
        project_root = args.project_root.resolve()
        config_path = args.config
        if config_path is None:
            config_path = project_root / "config" / "groups.config.yml"
        elif not config_path.is_absolute():
            config_path = (project_root / config_path).resolve()
        try:
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
                build_pipeline_fn=build_pipeline,
                clean_mod=clean_mod,
                load_config_fn=load_config,
            )
        except (FeatureError, ValueError, FileNotFoundError) as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

    if args.command == "ngram":
        try:
            if args.trace is not None:
                return write_ngrams_from_trace(
                    trace_path=args.trace,
                    n=args.n,
                    field=args.field,
                    by_group=bool(args.by_group),
                    min_count=args.min_count,
                    top=args.top,
                    output_format=args.format,
                    out_path=args.out,
                )

            project_root = args.project_root.resolve()
            config_path = args.config
            if config_path is None:
                config_path = project_root / "config" / "groups.config.yml"
            elif not config_path.is_absolute():
                config_path = (project_root / config_path).resolve()

            return write_ngrams_from_config(
                project_root=project_root,
                config_path=config_path,
                n=args.n,
                field=args.field,
                by_group=bool(args.by_group),
                min_count=args.min_count,
                top=args.top,
                output_format=args.format,
                out_path=args.out,
            )
        except NgramError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
