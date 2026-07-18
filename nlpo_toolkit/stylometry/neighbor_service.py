from __future__ import annotations

from .neighbor_models import NeighborRankingRequest
from .neighbor_results import NeighborRankingResult
from .ports import StylometryCommandDependencies
from .ranking import build_neighbor_rankings
from .standardization import fit_zscore_model, transform_feature_dataset


def execute_neighbor_ranking(
    request: NeighborRankingRequest,
    *,
    dependencies: StylometryCommandDependencies,
) -> NeighborRankingResult:
    dataset = dependencies.read_feature_dataset(
        request.features_path,
        input_format=request.input_format,
        selection=request.selection,
    )
    model = fit_zscore_model(dataset)
    standardized = transform_feature_dataset(dataset, model=model)
    return NeighborRankingResult(
        metric=request.metric,
        rankings=build_neighbor_rankings(
            standardized, metric=request.metric, top=request.top
        ),
        input_feature_names=model.input_feature_names,
        retained_feature_names=model.retained_feature_names,
        dropped_zero_variance_features=model.dropped_zero_variance_features,
    )
