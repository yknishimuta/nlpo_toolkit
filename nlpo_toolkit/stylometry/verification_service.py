from __future__ import annotations

from .authorship import build_work_profiles, label_feature_dataset
from .ports import StylometryCommandDependencies
from .verification import evaluate_verification
from .verification_models import VerificationRequest
from .verification_results import VerificationResult


def execute_verification(
    request: VerificationRequest, *, dependencies: StylometryCommandDependencies
) -> VerificationResult:
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
    return evaluate_verification(
        labeled.feature_names,
        build_work_profiles(labeled),
        candidate_author=request.candidate_author,
        query_work=request.query_work,
        settings=request.thresholds,
    )
