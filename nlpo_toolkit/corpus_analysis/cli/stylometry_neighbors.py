from __future__ import annotations

import argparse
from pathlib import Path

from nlpo_toolkit.stylometry.composition import default_stylometry_dependencies
from nlpo_toolkit.stylometry.errors import StylometryError
from nlpo_toolkit.stylometry.metrics import StylometryMetric
from nlpo_toolkit.stylometry.models import FeatureSelection
from nlpo_toolkit.stylometry.neighbor_models import NeighborRankingRequest
from nlpo_toolkit.stylometry.neighbor_service import execute_neighbor_ranking

from .common import CLIContext, set_handler
from .output import open_cli_output, present_error
from .stylometry_neighbor_rendering import write_neighbor_result


def register_neighbors(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("neighbors")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--input-format", choices=("csv", "tsv"), default="csv")
    parser.add_argument("--id-column", default="group")
    parser.add_argument("--feature-prefix", action="append", default=[])
    parser.add_argument("--feature-column", action="append", default=[])
    parser.add_argument(
        "--metric",
        choices=tuple(metric.value for metric in StylometryMetric),
        default=StylometryMetric.BURROWS_DELTA.value,
    )
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--format", choices=("csv", "tsv"), default="csv")
    set_handler(parser, execute_neighbors)


def execute_neighbors(args: argparse.Namespace, context: CLIContext) -> int:
    try:
        request = NeighborRankingRequest(
            features_path=args.features.expanduser().resolve(),
            input_format=args.input_format,
            selection=FeatureSelection(
                id_column=args.id_column,
                prefixes=tuple(args.feature_prefix),
                columns=tuple(args.feature_column),
            ),
            metric=StylometryMetric(args.metric),
            top=args.top,
        )
        result = execute_neighbor_ranking(
            request,
            dependencies=default_stylometry_dependencies(),
        )
        with open_cli_output(path=args.out, stdout=context.stdout) as stream:
            write_neighbor_result(result, stream=stream, output_format=args.format)
        if result.dropped_feature_count:
            print(
                "[STYLOMETRY] excluded zero-variance features: "
                f"{result.dropped_feature_count}",
                file=context.stderr,
            )
        return 0
    except (StylometryError, OSError) as exc:
        present_error(exc, stderr=context.stderr)
        return 1
