from __future__ import annotations

import argparse
from pathlib import Path

from nlpo_toolkit.stylometry.composition import default_stylometry_dependencies
from nlpo_toolkit.stylometry.errors import StylometryError
from nlpo_toolkit.stylometry.evaluation_models import LeaveOneWorkOutEvaluationRequest
from nlpo_toolkit.stylometry.evaluation_service import execute_lowo_evaluation
from nlpo_toolkit.stylometry.models import FeatureSelection

from .common import CLIContext, set_handler
from .output import open_cli_output, present_error
from .stylometry_evaluation_rendering import write_lowo_folds, write_lowo_summary_json


def register_evaluate_lowo(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("evaluate-lowo")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--input-format", choices=("csv", "tsv"), default="csv")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-format", choices=("csv", "tsv"), default="csv")
    parser.add_argument("--id-column", default="group")
    parser.add_argument("--metadata-id-column", default=None)
    parser.add_argument("--author-column", default="author")
    parser.add_argument("--work-column", default="work")
    parser.add_argument("--feature-prefix", action="append", default=[])
    parser.add_argument("--feature-column", action="append", default=[])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--format", choices=("csv", "tsv"), default="csv")
    parser.add_argument("--summary-out", type=Path, default=None)
    set_handler(parser, execute_evaluate_lowo)


def execute_evaluate_lowo(args: argparse.Namespace, context: CLIContext) -> int:
    try:
        output = args.out.expanduser().resolve() if args.out is not None else None
        summary_output = (
            args.summary_out.expanduser().resolve()
            if args.summary_out is not None
            else None
        )
        if output is not None and output == summary_output:
            raise StylometryError("--out and --summary-out must be different paths")
        request = LeaveOneWorkOutEvaluationRequest(
            features_path=args.features.expanduser().resolve(),
            input_format=args.input_format,
            feature_selection=FeatureSelection(
                id_column=args.id_column,
                prefixes=tuple(args.feature_prefix),
                columns=tuple(args.feature_column),
            ),
            metadata_path=args.metadata.expanduser().resolve(),
            metadata_format=args.metadata_format,
            metadata_id_column=args.metadata_id_column or args.id_column,
            author_column=args.author_column,
            work_column=args.work_column,
        )
        result = execute_lowo_evaluation(
            request, dependencies=default_stylometry_dependencies()
        )
        with open_cli_output(path=output, stdout=context.stdout) as stream:
            write_lowo_folds(result, stream=stream, output_format=args.format)
        if summary_output is not None:
            write_lowo_summary_json(result, path=summary_output)
        summary = result.summary
        print(
            f"[STYLOMETRY] LOWO work accuracy: {summary.correct_work_count}/"
            f"{summary.work_count} ({summary.accuracy})",
            file=context.stderr,
        )
        print(
            f"[STYLOMETRY] LOWO macro author accuracy: {summary.macro_author_accuracy}",
            file=context.stderr,
        )
        return 0
    except (StylometryError, OSError) as exc:
        present_error(exc, stderr=context.stderr)
        return 1
