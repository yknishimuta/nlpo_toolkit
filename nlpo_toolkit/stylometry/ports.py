from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import FeatureDataset, FeatureSelection, InputFormat


class FeatureDatasetReader(Protocol):
    def __call__(
        self,
        path: Path,
        *,
        input_format: InputFormat,
        selection: FeatureSelection,
    ) -> FeatureDataset: ...


@dataclass(frozen=True)
class StylometryCommandDependencies:
    read_feature_dataset: FeatureDatasetReader
