from __future__ import annotations

from .csv_reader import read_feature_dataset
from .ports import StylometryCommandDependencies
from .metadata_reader import read_authorship_metadata


def default_stylometry_dependencies() -> StylometryCommandDependencies:
    return StylometryCommandDependencies(
        read_feature_dataset=read_feature_dataset,
        read_authorship_metadata=read_authorship_metadata,
    )
