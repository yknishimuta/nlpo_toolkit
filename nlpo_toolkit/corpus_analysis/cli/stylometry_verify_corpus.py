from __future__ import annotations

import argparse
from pathlib import Path

from nlpo_toolkit.stylometry.composition import default_stylometry_dependencies
from nlpo_toolkit.stylometry.errors import StylometryError

from ..composition import default_feature_command_dependencies
from ..features.corpus_verification_models import CorpusVerificationRequest
from ..features.corpus_verification_service import (
    CorpusVerificationDependencies,
    execute_corpus_verification,
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
from .stylometry_corpus_verification_rendering import (
    write_corpus_verification_json,
    write_corpus_verification_vocabulary_audit,
)
from .stylometry_verification_options import (
    add_verification_decision_arguments,
    build_threshold_settings,
)
from .stylometry_verification_rendering import write_verification_calibration


def register_verify_corpus(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("verify-corpus")
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
    add_verification_decision_arguments(parser)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--calibration-out", type=Path, default=None)
    parser.add_argument("--calibration-format", choices=("csv", "tsv"), default="csv")
    parser.add_argument("--vocabulary-audit-out", type=Path, default=None)
    add_feature_options(parser)
    add_grouping_override_arguments(parser)
    add_empty_group_argument(parser)
    set_handler(parser, execute_verify_corpus)


def execute_verify_corpus(args: argparse.Namespace, context: CLIContext) -> int:
    try:
        output_paths = tuple(
            path.expanduser().resolve()
            for path in (args.out, args.calibration_out, args.vocabulary_audit_out)
            if path is not None
        )
        if len(output_paths) != len(set(output_paths)):
            raise FeatureError(
                "verification, calibration, and vocabulary audit outputs must differ"
            )
        features = build_feature_request(
            args, corpus=build_corpus_preparation_request(args)
        )
        request = CorpusVerificationRequest(
            features=features,
            metadata_path=args.metadata.expanduser().resolve(),
            candidate_author=args.candidate_author,
            query_work=args.query_work,
            thresholds=build_threshold_settings(args),
            metadata_format=args.metadata_format,
            metadata_group_column=args.metadata_group_column,
            author_column=args.author_column,
            work_column=args.work_column,
        )
        stylometry = default_stylometry_dependencies()
        result = execute_corpus_verification(
            request,
            dependencies=CorpusVerificationDependencies(
                default_feature_command_dependencies(),
                stylometry.read_authorship_metadata,
            ),
        )
        with open_cli_output(path=args.out, stdout=context.stdout) as stream:
            write_corpus_verification_json(result, stream=stream)
        if args.calibration_out is not None:
            write_verification_calibration(
                result.verification,
                path=args.calibration_out.expanduser().resolve(),
                output_format=args.calibration_format,
            )
        if args.vocabulary_audit_out is not None:
            write_corpus_verification_vocabulary_audit(
                result,
                path=args.vocabulary_audit_out.expanduser().resolve(),
            )
        print(
            f"[STYLOMETRY] verification decision: {result.verification.decision.value}; "
            f"selected features: {result.selected_feature_count}; "
            f"vocabulary sha256: {result.vocabulary.sha256}",
            file=context.stderr,
        )
        return 0
    except (FeatureError, StylometryError, OSError, ValueError) as exc:
        present_error(exc, stderr=context.stderr)
        return 1
