from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import StylometryError
from .metrics import StylometryMetric


@dataclass(frozen=True)
class NeighborScore:
    neighbor_id: str
    score: float

    def __post_init__(self) -> None:
        if not self.neighbor_id.strip():
            raise StylometryError("neighbor identifier must not be empty")
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(self.score)
        ):
            raise StylometryError("neighbor score must be finite")
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True)
class ObservationNeighborRanking:
    query_id: str
    neighbors: tuple[NeighborScore, ...]

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise StylometryError("query identifier must not be empty")
        object.__setattr__(self, "neighbors", tuple(self.neighbors))
        identifiers = tuple(item.neighbor_id for item in self.neighbors)
        if self.query_id in identifiers or len(identifiers) != len(set(identifiers)):
            raise StylometryError("neighbor ranking contains an invalid candidate")


@dataclass(frozen=True)
class NeighborRankingResult:
    metric: StylometryMetric
    rankings: tuple[ObservationNeighborRanking, ...]
    input_feature_names: tuple[str, ...]
    retained_feature_names: tuple[str, ...]
    dropped_zero_variance_features: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.metric, StylometryMetric):
            raise StylometryError("neighbor result metric must be a StylometryMetric")
        for name in (
            "rankings",
            "input_feature_names",
            "retained_feature_names",
            "dropped_zero_variance_features",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def query_count(self) -> int:
        return len(self.rankings)

    @property
    def retained_feature_count(self) -> int:
        return len(self.retained_feature_names)

    @property
    def dropped_feature_count(self) -> int:
        return len(self.dropped_zero_variance_features)

    @property
    def output_row_count(self) -> int:
        return sum(len(item.neighbors) for item in self.rankings)
