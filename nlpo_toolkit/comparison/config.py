from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictInt, model_validator

from nlpo_toolkit.config_model import ConfigModel, PositiveFiniteFloat, StrippedNonBlankStr


ComparisonSortBy = Literal["log_likelihood", "abs_log_ratio", "total_count", "item"]
PositiveStrictInt = Annotated[StrictInt, Field(gt=0)]


class ComparisonSortConfig(ConfigModel):
    by: ComparisonSortBy = "log_likelihood"
    descending: StrictBool = True


class ComparisonSpec(ConfigModel):
    name: StrippedNonBlankStr
    group_a: StrippedNonBlankStr
    group_b: StrippedNonBlankStr
    scale: PositiveStrictInt = 10_000
    zero_correction: PositiveFiniteFloat = 0.5
    min_total_count: PositiveStrictInt = 1
    sort: ComparisonSortConfig = Field(default_factory=ComparisonSortConfig)

    @model_validator(mode="after")
    def validate_distinct_groups(self) -> ComparisonSpec:
        if self.group_a == self.group_b:
            raise ValueError("group_a and group_b must be different")
        return self
