from __future__ import annotations

import argparse
from pathlib import Path

from nlpo_toolkit.stylometry.composition import default_stylometry_dependencies
from nlpo_toolkit.stylometry.errors import StylometryError

from ..composition import default_feature_command_dependencies
from ..features.corpus_verification_evaluation_models import (
    CorpusVerificationEvaluationRequest,
)
from ..features.corpus_verification_evaluation_service import (
    CorpusVerificationEvaluationDependencies,
    execute_corpus_verification_evaluation,
)
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
from .stylometry_verification_evaluation_rendering import (
    write_verification_evaluation_calibration,
    write_verification_evaluation_folds,
    write_verification_evaluation_summary,
    write_verification_evaluation_vocabulary_audit,
)
from .stylometry_verification_options import build_threshold_settings


def register_evaluate_verification_corpus(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("evaluate-verification-corpus")
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
    parser.add_argument("--candidate-author", required=True)
    parser.add_argument("--genuine-quantile", type=float, default=0.95)
    parser.add_argument("--impostor-quantile", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--format", choices=("csv", "tsv"), default="csv")
    parser.add_argument("--vocabulary-audit-out", type=Path, default=None)
    parser.add_argument("--calibration-out", type=Path, default=None)
    parser.add_argument("--calibration-format", choices=("csv", "tsv"), default="csv")
    add_feature_options(parser)
    add_grouping_override_arguments(parser)
    add_empty_group_argument(parser)
    set_handler(parser, execute_evaluate_verification_corpus)


def execute_evaluate_verification_corpus(
    args: argparse.Namespace, context: CLIContext
) -> int:
    try:
        outputs = tuple(
            path.expanduser().resolve()
            for path in (args.out, args.summary_out, args.vocabulary_audit_out, args.calibration_out)
            if path is not None
        )
        inputs = (
            args.metadata.expanduser().resolve(),
            build_corpus_preparation_request(args).config_path,
        )
        if len(outputs) != len(set(outputs)) or any(path in inputs for path in outputs):
            raise FeatureError("verification evaluation outputs must all use different paths")
        corpus_request = build_corpus_preparation_request(args)
        request = CorpusVerificationEvaluationRequest(
            build_feature_request(args, corpus=corpus_request),
            args.metadata.expanduser().resolve(),
            args.candidate_author,
            build_threshold_settings(args),
            args.metadata_format,
            args.metadata_group_column,
            args.author_column,
            args.work_column,
        )
        stylometry = default_stylometry_dependencies()
        result = execute_corpus_verification_evaluation(
            request,
            dependencies=CorpusVerificationEvaluationDependencies(
                default_feature_command_dependencies(),
                stylometry.read_authorship_metadata,
            ),
        )
        with open_cli_output(path=args.out, stdout=context.stdout) as stream:
            write_verification_evaluation_folds(result, stream=stream, output_format=args.format)
        write_verification_evaluation_summary(result, path=args.summary_out.expanduser().resolve())
        if args.vocabulary_audit_out is not None:
            write_verification_evaluation_vocabulary_audit(result, path=args.vocabulary_audit_out.expanduser().resolve())
        if args.calibration_out is not None:
            write_verification_evaluation_calibration(result, path=args.calibration_out.expanduser().resolve(), output_format=args.calibration_format)
        summary = result.summary
        print(f"[STYLOMETRY] verification evaluation folds: {len(result.folds)}", file=context.stderr)
        print(f"[STYLOMETRY] coverage: {summary.coverage}", file=context.stderr)
        print(f"[STYLOMETRY] decisive accuracy: {summary.decisive_accuracy}", file=context.stderr)
        print(f"[STYLOMETRY] false accept rate: {summary.false_accept_rate}", file=context.stderr)
        print(f"[STYLOMETRY] false reject rate: {summary.false_reject_rate}", file=context.stderr)
        return 0
    except (FeatureError, StylometryError, OSError, ValueError) as exc:
        present_error(exc, stderr=context.stderr)
        return 1
