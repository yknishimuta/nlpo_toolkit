from __future__ import annotations

import argparse
from pathlib import Path

from nlpo_toolkit.stylometry.composition import default_stylometry_dependencies
from nlpo_toolkit.stylometry.errors import StylometryError
from nlpo_toolkit.stylometry.models import BurrowsDeltaRequest, FeatureSelection
from nlpo_toolkit.stylometry.service import execute_burrows_delta

from .common import CLIContext, set_handler
from .output import open_cli_output, present_error
from .stylometry_rendering import write_burrows_delta_result
from .stylometry_evaluate_lowo import register_evaluate_lowo
from .stylometry_neighbors import register_neighbors


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("stylometry")
    commands = parser.add_subparsers(dest="stylometry_command", required=True)
    register_evaluate_lowo(commands)
    register_neighbors(commands)
    delta = commands.add_parser("delta")
    delta.add_argument("--features", type=Path, required=True)
    delta.add_argument("--input-format", choices=("csv", "tsv"), default="csv")
    delta.add_argument("--id-column", default="group")
    delta.add_argument("--feature-prefix", action="append", default=[])
    delta.add_argument("--feature-column", action="append", default=[])
    delta.add_argument("--out", type=Path, default=None)
    delta.add_argument("--format", choices=("csv", "tsv"), default="csv")
    set_handler(delta, execute_delta)


def execute_delta(args: argparse.Namespace, context: CLIContext) -> int:
    try:
        request = BurrowsDeltaRequest(
            features_path=args.features.expanduser().resolve(),
            input_format=args.input_format,
            selection=FeatureSelection(
                id_column=args.id_column,
                prefixes=tuple(args.feature_prefix),
                columns=tuple(args.feature_column),
            ),
        )
        result = execute_burrows_delta(
            request,
            dependencies=default_stylometry_dependencies(),
        )
        with open_cli_output(path=args.out, stdout=context.stdout) as stream:
            write_burrows_delta_result(
                result,
                stream=stream,
                output_format=args.format,
            )
        dropped = len(result.dropped_zero_variance_features)
        if dropped:
            print(
                f"[STYLOMETRY] excluded zero-variance features: {dropped}",
                file=context.stderr,
            )
        return 0
    except (StylometryError, OSError) as exc:
        present_error(exc, stderr=context.stderr)
        return 1
