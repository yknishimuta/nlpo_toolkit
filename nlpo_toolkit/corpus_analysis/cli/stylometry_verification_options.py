from __future__ import annotations

import argparse
from pathlib import Path

from nlpo_toolkit.stylometry.models import FeatureSelection
from nlpo_toolkit.stylometry.verification_models import VerificationThresholdSettings


def add_verification_input_arguments(parser: argparse.ArgumentParser) -> None:
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


def build_feature_selection(args: argparse.Namespace) -> FeatureSelection:
    return FeatureSelection(
        id_column=args.id_column,
        prefixes=tuple(args.feature_prefix),
        columns=tuple(args.feature_column),
    )


def build_threshold_settings(
    args: argparse.Namespace,
) -> VerificationThresholdSettings:
    return VerificationThresholdSettings(args.genuine_quantile, args.impostor_quantile)
