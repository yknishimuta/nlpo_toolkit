from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import FeatureDataset, FeatureSelection, InputFormat
from .evaluation_models import AuthorshipMetadata


class FeatureDatasetReader(Protocol):
    def __call__(
        self,
        path: Path,
        *,
        input_format: InputFormat,
        selection: FeatureSelection,
    ) -> FeatureDataset: ...


class AuthorshipMetadataReader(Protocol):
    def __call__(
        self,
        path: Path,
        *,
        input_format: InputFormat,
        id_column: str,
        author_column: str,
        work_column: str,
    ) -> AuthorshipMetadata: ...


@dataclass(frozen=True)
class StylometryCommandDependencies:
    read_feature_dataset: FeatureDatasetReader
    read_authorship_metadata: AuthorshipMetadataReader
