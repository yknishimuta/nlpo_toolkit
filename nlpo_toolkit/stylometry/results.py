from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import StylometryError


@dataclass(frozen=True)
class DeltaPair:
    sample_a: str
    sample_b: str
    distance: float

    def __post_init__(self) -> None:
        if not self.sample_a or not self.sample_b:
            raise StylometryError("Delta pair identifiers must not be empty")
        if not math.isfinite(self.distance) or self.distance < 0.0:
            raise StylometryError("Burrows's Delta must be finite and non-negative")


@dataclass(frozen=True)
class BurrowsDeltaResult:
    pairs: tuple[DeltaPair, ...]
    input_feature_names: tuple[str, ...]
    retained_feature_names: tuple[str, ...]
    dropped_zero_variance_features: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "pairs",
            "input_feature_names",
            "retained_feature_names",
            "dropped_zero_variance_features",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
