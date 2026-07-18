from __future__ import annotations

import csv
from typing import TextIO

from nlpo_toolkit.stylometry.neighbor_results import NeighborRankingResult


NEIGHBOR_COLUMNS = (
    "query_id",
    "rank",
    "neighbor_id",
    "metric",
    "score",
)


def write_neighbor_result(
    result: NeighborRankingResult,
    *,
    stream: TextIO,
    output_format: str,
) -> None:
    writer = csv.writer(stream, delimiter="," if output_format == "csv" else "\t")
    writer.writerow(NEIGHBOR_COLUMNS)
    for ranking in result.rankings:
        for rank, neighbor in enumerate(ranking.neighbors, start=1):
            writer.writerow(
                (
                    ranking.query_id,
                    rank,
                    neighbor.neighbor_id,
                    result.metric.value,
                    neighbor.score,
                )
            )
