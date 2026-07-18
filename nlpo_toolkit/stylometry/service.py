from __future__ import annotations

from .delta import build_delta_pairs
from .models import BurrowsDeltaRequest
from .ports import StylometryCommandDependencies
from .results import BurrowsDeltaResult
from .standardization import fit_zscore_model, transform_feature_dataset


def execute_burrows_delta(
    request: BurrowsDeltaRequest,
    *,
    dependencies: StylometryCommandDependencies,
) -> BurrowsDeltaResult:
    dataset = dependencies.read_feature_dataset(
        request.features_path,
        input_format=request.input_format,
        selection=request.selection,
    )
    model = fit_zscore_model(dataset)
    standardized = transform_feature_dataset(dataset, model=model)
    return BurrowsDeltaResult(
        pairs=build_delta_pairs(standardized),
        input_feature_names=model.input_feature_names,
        retained_feature_names=model.retained_feature_names,
        dropped_zero_variance_features=model.dropped_zero_variance_features,
    )
