from __future__ import annotations

from .authorship import label_feature_dataset
from .ports import StylometryCommandDependencies
from .stability_engine import evaluate_verification_stability
from .stability_models import VerificationStabilityRequest
from .stability_results import VerificationStabilityResult


def execute_verification_stability(
    request: VerificationStabilityRequest,
    *,
    dependencies: StylometryCommandDependencies,
) -> VerificationStabilityResult:
    dataset = dependencies.read_feature_dataset(
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
    labeled = label_feature_dataset(dataset, metadata=metadata)
    return evaluate_verification_stability(
        labeled,
        candidate_author=request.candidate_author,
        query_work=request.query_work,
        verification_thresholds=request.verification_thresholds,
        settings=request.stability,
    )
