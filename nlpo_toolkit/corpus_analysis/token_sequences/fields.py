from __future__ import annotations

from typing import Literal

from .models import SequenceItem


TokenField = Literal["token", "lemma"]


def token_field_value(item: SequenceItem, field: TokenField) -> str:
    if field == "token":
        return item.token
    if field == "lemma":
        return item.lemma or ""
    raise ValueError("field must be 'token' or 'lemma'.")
