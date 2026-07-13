from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


GroupingOverride = Literal["per_file", "auto_single_cleaned"]


@dataclass(frozen=True)
class CorpusPreparationRequest:
    project_root: Path
    config_path: Path
    grouping_override: GroupingOverride | None = None
    error_on_empty_group: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.project_root, Path):
            raise TypeError("project_root must be a Path")
        if not isinstance(self.config_path, Path):
            raise TypeError("config_path must be a Path")
        if self.grouping_override not in {None, "per_file", "auto_single_cleaned"}:
            raise ValueError(
                "grouping_override must be 'per_file', 'auto_single_cleaned', or None"
            )
        if not isinstance(self.error_on_empty_group, bool):
            raise TypeError("error_on_empty_group must be a bool")
