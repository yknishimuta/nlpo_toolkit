from __future__ import annotations

from .errors import StylometryMetricError
from .metrics import (
    ScorePreference,
    StylometryMetric,
    compute_stylometry_score,
    score_preference,
)
from .models import StandardizedFeatureDataset
from .neighbor_results import NeighborScore, ObservationNeighborRanking


def build_neighbor_rankings(
    dataset: StandardizedFeatureDataset,
    *,
    metric: StylometryMetric,
    top: int | None = None,
) -> tuple[ObservationNeighborRanking, ...]:
    if len(dataset.observations) < 2:
        raise StylometryMetricError(
            "nearest-neighbor ranking requires at least two observations"
        )
    if top is not None and (
        isinstance(top, bool) or not isinstance(top, int) or top < 1
    ):
        raise StylometryMetricError("top must be a positive integer")
    preference = score_preference(metric)
    rankings = []
    for query in dataset.observations:
        scores = []
        for candidate in dataset.observations:
            if candidate.identifier == query.identifier:
                continue
            try:
                score = compute_stylometry_score(
                    query.values, candidate.values, metric=metric
                )
            except StylometryMetricError as exc:
                if metric in (
                    StylometryMetric.COSINE_DELTA,
                    StylometryMetric.COSINE_SIMILARITY,
                ):
                    zero_id = (
                        query.identifier
                        if not any(query.values)
                        else candidate.identifier
                    )
                    raise StylometryMetricError(
                        f"cosine metric is undefined for observation {zero_id!r}: "
                        "standardized vector has zero norm"
                    ) from exc
                raise
            scores.append(NeighborScore(candidate.identifier, score))
        if preference is ScorePreference.LOWER_IS_BETTER:
            scores.sort(key=lambda item: (item.score, item.neighbor_id))
        else:
            scores.sort(key=lambda item: (-item.score, item.neighbor_id))
        rankings.append(
            ObservationNeighborRanking(
                query.identifier,
                tuple(scores if top is None else scores[:top]),
            )
        )
    return tuple(rankings)
