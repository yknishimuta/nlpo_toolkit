from __future__ import annotations

from .errors import StylometryError
from .metrics import burrows_delta as burrows_delta_values
from .models import StandardizedFeatureDataset, StandardizedObservation
from .results import DeltaPair


def burrows_delta(
    first: StandardizedObservation,
    second: StandardizedObservation,
) -> float:
    try:
        return burrows_delta_values(first.values, second.values)
    except StylometryError as exc:
        raise StylometryError(
            "standardized vectors must have the same non-zero width"
        ) from exc


def build_delta_pairs(dataset: StandardizedFeatureDataset) -> tuple[DeltaPair, ...]:
    pairs = tuple(
        DeltaPair(
            sample_a=first.identifier,
            sample_b=second.identifier,
            distance=burrows_delta(first, second),
        )
        for index, first in enumerate(dataset.observations)
        for second in dataset.observations[index + 1 :]
    )
    return tuple(
        sorted(pairs, key=lambda item: (item.distance, item.sample_a, item.sample_b))
    )
