from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from nlpo_toolkit.cleaner_contracts import (
    CleanerApplicationError,
    CleanerExecutionRequest,
    CleanerExecutionResult,
)

from .config_loader import inspect_cleaner_config
from .service import execute_cleaner


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "config" / "sample.yml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean Latin corpus text files.")
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Cleaner YAML config path.",
    )
    return parser


def _present_result(result: CleanerExecutionResult) -> None:
    for file in result.files:
        print(f"[{result.kind}] cleaned: {file.input_path} -> {file.output_path}")
    print(
        f"[{result.kind}] cleaned_files={len(result.files)} "
        f"reference_events={result.reference_event_count}"
    )
    if result.ref_tsv_path is not None:
        print(f"[{result.kind}] reference_events_tsv={result.ref_tsv_path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inspection = inspect_cleaner_config(args.config)
        result = execute_cleaner(CleanerExecutionRequest(inspection=inspection))
    except CleanerApplicationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    _present_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
