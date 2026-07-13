from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from nlpo_toolkit.config_model import ConfigModel, StrippedNonBlankStr


class PartitionSpec(ConfigModel):
    name: StrippedNonBlankStr
    whole: StrippedNonBlankStr
    parts: tuple[StrippedNonBlankStr, ...] = Field(min_length=2)
    on_mismatch: Literal["warn", "error"] = "warn"
    report: Literal["mismatches", "all"] = "mismatches"

    @model_validator(mode="after")
    def validate_group_roles(self) -> PartitionSpec:
        if len(set(self.parts)) != len(self.parts):
            raise ValueError("parts must not contain duplicate group names")
        if self.whole in self.parts:
            raise ValueError("whole must not be included in parts")
        return self
