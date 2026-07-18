from __future__ import annotations

import argparse
from pathlib import Path

from nlpo_toolkit.stylometry.composition import default_stylometry_dependencies
from nlpo_toolkit.stylometry.errors import StylometryError
from nlpo_toolkit.stylometry.models import FeatureSelection
from nlpo_toolkit.stylometry.verification_models import (
    VerificationRequest,
    VerificationThresholdSettings,
)
from nlpo_toolkit.stylometry.verification_service import execute_verification

from .common import CLIContext, set_handler
from .output import open_cli_output, present_error
from .stylometry_verification_rendering import (
    write_verification_calibration,
    write_verification_json,
)


def register_verify(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("verify")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--input-format", choices=("csv", "tsv"), default="csv")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-format", choices=("csv", "tsv"), default="csv")
    parser.add_argument("--id-column", default="group")
    parser.add_argument("--metadata-id-column", default="sample_id")
    parser.add_argument("--author-column", default="author")
    parser.add_argument("--work-column", default="work")
    parser.add_argument("--feature-prefix", action="append", default=[])
    parser.add_argument("--feature-column", action="append", default=[])
    parser.add_argument("--candidate-author", required=True)
    parser.add_argument("--query-work", required=True)
    parser.add_argument("--genuine-quantile", type=float, default=0.95)
    parser.add_argument("--impostor-quantile", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--calibration-out", type=Path, default=None)
    parser.add_argument(
        "--calibration-format", choices=("csv", "tsv"), default="csv"
    )
    set_handler(parser, execute_verify)


def execute_verify(args: argparse.Namespace, context: CLIContext) -> int:
    try:
        output = args.out.expanduser().resolve() if args.out is not None else None
        calibration = (
            args.calibration_out.expanduser().resolve()
            if args.calibration_out is not None else None
        )
        if output is not None and output == calibration:
            raise StylometryError("--out and --calibration-out must be different paths")
        request = VerificationRequest(
            features_path=args.features.expanduser().resolve(),
            metadata_path=args.metadata.expanduser().resolve(),
            input_format=args.input_format,
            metadata_format=args.metadata_format,
            feature_selection=FeatureSelection(
                id_column=args.id_column,
                prefixes=tuple(args.feature_prefix),
                columns=tuple(args.feature_column),
            ),
            metadata_id_column=args.metadata_id_column,
            author_column=args.author_column,
            work_column=args.work_column,
            candidate_author=args.candidate_author,
            query_work=args.query_work,
            thresholds=VerificationThresholdSettings(
                args.genuine_quantile, args.impostor_quantile
            ),
        )
        result = execute_verification(
            request, dependencies=default_stylometry_dependencies()
        )
        with open_cli_output(path=output, stdout=context.stdout) as stream:
            write_verification_json(result, stream=stream)
        if calibration is not None:
            write_verification_calibration(
                result, path=calibration, output_format=args.calibration_format
            )
        return 0
    except (StylometryError, OSError) as exc:
        present_error(exc, stderr=context.stderr)
        return 1
