from __future__ import annotations

from .authorship import (
    build_work_profiles,
    label_feature_dataset,
    validate_lowo_dataset,
)
from .evaluation import evaluate_lowo
from .evaluation_models import LeaveOneWorkOutEvaluationRequest
from .evaluation_results import LeaveOneWorkOutEvaluationResult
from .ports import StylometryCommandDependencies


def execute_lowo_evaluation(
    request: LeaveOneWorkOutEvaluationRequest,
    *,
    dependencies: StylometryCommandDependencies,
) -> LeaveOneWorkOutEvaluationResult:
    features = dependencies.read_feature_dataset(
        request.features_path,
        input_format=request.input_format,
        selection=request.feature_selection,
    )
    metadata = dependencies.read_authorship_metadata(
        request.metadata_path,
        input_format=request.metadata_format,
        id_column=request.metadata_id_column,
        author_column=request.author_column,
        work_column=request.work_column,
    )
    labeled = label_feature_dataset(features, metadata=metadata)
    validate_lowo_dataset(labeled)
    profiles = build_work_profiles(labeled)
    return evaluate_lowo(labeled.feature_names, profiles)
