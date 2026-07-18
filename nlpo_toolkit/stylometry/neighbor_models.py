from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import StylometryError
from .metrics import StylometryMetric
from .models import FeatureSelection, InputFormat


@dataclass(frozen=True)
class NeighborRankingRequest:
    features_path: Path
    input_format: InputFormat
    selection: FeatureSelection
    metric: StylometryMetric = StylometryMetric.BURROWS_DELTA
    top: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.features_path, Path):
            raise StylometryError("features_path must be a Path")
        if self.input_format not in ("csv", "tsv"):
            raise StylometryError("input_format must be 'csv' or 'tsv'")
        if not isinstance(self.metric, StylometryMetric):
            raise StylometryError("metric must be a StylometryMetric")
        if self.top is not None and (
            isinstance(self.top, bool) or not isinstance(self.top, int) or self.top < 1
        ):
            raise StylometryError("top must be a positive integer")
