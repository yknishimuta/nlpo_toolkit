from __future__ import annotations

import math
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
)


class ConfigModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=True,
    )


def require_non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must be a non-empty string")
    return value


def strip_non_blank(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("must be a non-empty string")
    return stripped


def positive_finite_number(value: int | float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("must be a positive finite number")
    return number


NonBlankStr = Annotated[StrictStr, AfterValidator(require_non_blank)]
StrippedNonBlankStr = Annotated[StrictStr, AfterValidator(strip_non_blank)]
PositiveStrictIntNumber = Annotated[StrictInt, Field(gt=0)]
PositiveStrictFloatNumber = Annotated[StrictFloat, Field(gt=0)]
PositiveFiniteFloat = Annotated[
    PositiveStrictIntNumber | PositiveStrictFloatNumber,
    AfterValidator(positive_finite_number),
]
