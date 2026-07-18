from __future__ import annotations

import csv
from typing import TextIO

from nlpo_toolkit.stylometry.results import BurrowsDeltaResult


def write_burrows_delta_result(
    result: BurrowsDeltaResult,
    *,
    stream: TextIO,
    output_format: str,
) -> None:
    delimiter = "," if output_format == "csv" else "\t"
    writer = csv.writer(stream, delimiter=delimiter)
    writer.writerow(("sample_a", "sample_b", "burrows_delta"))
    for pair in result.pairs:
        writer.writerow((pair.sample_a, pair.sample_b, pair.distance))
