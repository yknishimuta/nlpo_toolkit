from __future__ import annotations

import argparse
from pathlib import Path

from nlpo_toolkit.stylometry.composition import default_stylometry_dependencies
from nlpo_toolkit.stylometry.errors import StylometryError

from ..composition import default_feature_command_dependencies
from ..features.corpus_lowo_models import CorpusLowoRequest
from ..features.corpus_lowo_service import CorpusLowoDependencies, execute_corpus_lowo
from ..features.errors import FeatureError
from .common import (
    CLIContext,
    add_empty_group_argument,
    add_grouping_override_arguments,
    add_project_config_arguments,
    build_corpus_preparation_request,
    set_handler,
)
from .feature_options import add_feature_options, build_feature_request
from .output import open_cli_output, present_error
from .stylometry_corpus_lowo_rendering import (
    write_corpus_lowo_folds,
    write_corpus_lowo_summary,
    write_vocabulary_audit,
)


def register_evaluate_lowo_corpus(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("evaluate-lowo-corpus")
    add_project_config_arguments(
        parser,
        project_root_help="Project root used to resolve corpus configuration.",
        config_help="Corpus configuration path.",
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-format", choices=("csv", "tsv"), default="csv")
    parser.add_argument("--metadata-group-column", default="group")
    parser.add_argument("--author-column", default="author")
    parser.add_argument("--work-column", default="work")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--format", choices=("csv", "tsv"), default="csv")
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--vocabulary-audit-out", type=Path, default=None)
    add_feature_options(parser)
    add_grouping_override_arguments(parser)
    add_empty_group_argument(parser)
    set_handler(parser, execute_evaluate_lowo_corpus)


def execute_evaluate_lowo_corpus(args: argparse.Namespace, context: CLIContext) -> int:
    try:
        outputs = tuple(
            path.expanduser().resolve()
            for path in (args.out, args.summary_out, args.vocabulary_audit_out)
            if path is not None
        )
        if len(outputs) != len(set(outputs)):
            raise FeatureError(
                "fold, summary, and vocabulary audit outputs must differ"
            )
        feature_request = build_feature_request(
            args, corpus=build_corpus_preparation_request(args)
        )
        request = CorpusLowoRequest(
            feature_request,
            args.metadata.expanduser().resolve(),
            args.metadata_format,
            args.metadata_group_column,
            args.author_column,
            args.work_column,
        )
        stylometry = default_stylometry_dependencies()
        result = execute_corpus_lowo(
            request,
            dependencies=CorpusLowoDependencies(
                default_feature_command_dependencies(),
                stylometry.read_authorship_metadata,
            ),
        )
        with open_cli_output(path=args.out, stdout=context.stdout) as stream:
            write_corpus_lowo_folds(result, stream=stream, output_format=args.format)
        if args.summary_out is not None:
            write_corpus_lowo_summary(
                result, path=args.summary_out.expanduser().resolve()
            )
        if args.vocabulary_audit_out is not None:
            write_vocabulary_audit(
                result, path=args.vocabulary_audit_out.expanduser().resolve()
            )
        print(
            f"[STYLOMETRY] LOWO work accuracy: {result.summary.correct_work_count}/"
            f"{result.summary.work_count} ({result.summary.accuracy})",
            file=context.stderr,
        )
        return 0
    except (FeatureError, StylometryError, OSError, ValueError) as exc:
        present_error(exc, stderr=context.stderr)
        return 1
