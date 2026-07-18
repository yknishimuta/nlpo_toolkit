from __future__ import annotations

import math

from .errors import StylometryError
from .models import (
    FeatureDataset,
    StandardizedFeatureDataset,
    StandardizedObservation,
    ZScoreModel,
)


def fit_zscore_model(dataset: FeatureDataset) -> ZScoreModel:
    if dataset.sample_count < 2:
        raise StylometryError("Burrows's Delta requires at least two observations")
    means: list[float] = []
    standard_deviations: list[float] = []
    retained_names: list[str] = []
    retained_indices: list[int] = []
    dropped: list[str] = []
    for index, name in enumerate(dataset.feature_names):
        values = tuple(item.values[index] for item in dataset.observations)
        mean = sum(values) / dataset.sample_count
        variance = sum((value - mean) ** 2 for value in values) / (
            dataset.sample_count - 1
        )
        deviation = math.sqrt(variance)
        if deviation == 0.0:
            dropped.append(name)
            continue
        retained_names.append(name)
        retained_indices.append(index)
        means.append(float(mean))
        standard_deviations.append(float(deviation))
    if not retained_names:
        raise StylometryError("all selected features have zero variance")
    return ZScoreModel(
        input_feature_names=dataset.feature_names,
        retained_feature_names=tuple(retained_names),
        retained_indices=tuple(retained_indices),
        dropped_zero_variance_features=tuple(dropped),
        means=tuple(means),
        standard_deviations=tuple(standard_deviations),
    )


def transform_feature_dataset(
    dataset: FeatureDataset,
    *,
    model: ZScoreModel,
) -> StandardizedFeatureDataset:
    if dataset.feature_names != model.input_feature_names:
        raise StylometryError("feature dataset schema does not match z-score model")
    if any(deviation <= 0.0 for deviation in model.standard_deviations):
        raise StylometryError("z-score model standard deviations must be positive")
    observations = tuple(
        StandardizedObservation(
            identifier=item.identifier,
            values=tuple(
                (item.values[index] - mean) / deviation
                for index, mean, deviation in zip(
                    model.retained_indices,
                    model.means,
                    model.standard_deviations,
                    strict=True,
                )
            ),
        )
        for item in dataset.observations
    )
    return StandardizedFeatureDataset(model.retained_feature_names, observations)
