from __future__ import annotations

import argparse
from pathlib import Path

from nlpo_toolkit.stylometry.composition import default_stylometry_dependencies
from nlpo_toolkit.stylometry.errors import StylometryError
from nlpo_toolkit.stylometry.stability_models import (
    ResamplingAxis,
    ResamplingIntervalSettings,
    VerificationStabilityRequest,
    VerificationStabilitySettings,
)
from nlpo_toolkit.stylometry.stability_service import execute_verification_stability

from .common import CLIContext, set_handler
from .output import present_error
from .stylometry_stability_rendering import (
    write_feature_stability,
    write_replicates,
    write_stability_json,
    write_stability_json_path,
)
from .stylometry_verification_options import (
    add_verification_input_arguments,
    build_feature_selection,
    build_threshold_settings,
)


def register_verify_stability(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("verify-stability")
    add_verification_input_arguments(parser)
    parser.add_argument(
        "--resample-axis",
        choices=tuple(item.value for item in ResamplingAxis),
        action="append",
        required=True,
    )
    parser.add_argument("--work-fraction", type=float, default=0.8)
    parser.add_argument("--feature-fraction", type=float, default=0.8)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--stability-threshold", type=float, default=0.8)
    parser.add_argument("--interval-lower", type=float, default=0.025)
    parser.add_argument("--interval-upper", type=float, default=0.975)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--replicates-out", type=Path, default=None)
    parser.add_argument("--replicates-format", choices=("csv", "tsv"), default="csv")
    parser.add_argument("--feature-stability-out", type=Path, default=None)
    set_handler(parser, execute_verify_stability)


def _path(value: Path | None) -> Path | None:
    return value.expanduser().resolve() if value is not None else None


def execute_verify_stability(args: argparse.Namespace, context: CLIContext) -> int:
    try:
        output = _path(args.out)
        replicates_output = _path(args.replicates_out)
        feature_output = _path(args.feature_stability_out)
        features = args.features.expanduser().resolve()
        metadata = args.metadata.expanduser().resolve()
        outputs = tuple(
            item
            for item in (output, replicates_output, feature_output)
            if item is not None
        )
        if len(outputs) != len(set(outputs)) or any(
            item in (features, metadata) for item in outputs
        ):
            raise StylometryError(
                "stability input and output paths must all be different"
            )
        axes = tuple(ResamplingAxis(value) for value in args.resample_axis)
        request = VerificationStabilityRequest(
            features,
            metadata,
            args.input_format,
            args.metadata_format,
            build_feature_selection(args),
            args.metadata_id_column,
            args.author_column,
            args.work_column,
            args.candidate_author,
            args.query_work,
            build_threshold_settings(args),
            VerificationStabilitySettings(
                axes,
                args.iterations,
                args.seed,
                args.max_attempts,
                args.work_fraction,
                args.feature_fraction,
                args.stability_threshold,
                ResamplingIntervalSettings(args.interval_lower, args.interval_upper),
            ),
        )
        result = execute_verification_stability(
            request, dependencies=default_stylometry_dependencies()
        )
        if output is None:
            write_stability_json(result, stream=context.stdout)
        else:
            write_stability_json_path(result, path=output)
        if replicates_output is not None:
            write_replicates(
                result, path=replicates_output, output_format=args.replicates_format
            )
        if feature_output is not None:
            write_feature_stability(result, path=feature_output, output_format="csv")
        summary = result.decision_stability
        print(
            f"Base verification decision: {result.base_result.decision.value}",
            file=context.stderr,
        )
        print(
            f"Resampling iterations: {result.successful_iterations} successful / "
            f"{result.attempted_iterations} attempts",
            file=context.stderr,
        )
        print(
            f"Decision rates: accept={summary.accept_rate}, "
            f"inconclusive={summary.inconclusive_rate}, reject={summary.reject_rate}",
            file=context.stderr,
        )
        print(
            f"Modal decision: {summary.modal_decision.value} ({summary.modal_decision_rate})",
            file=context.stderr,
        )
        print(
            f"Verification stability: {summary.status.value} "
            f"(threshold={result.settings.stability_threshold})",
            file=context.stderr,
        )
        return 0
    except (StylometryError, OSError) as exc:
        present_error(exc, stderr=context.stderr)
        return 1
