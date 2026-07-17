from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Generic, Protocol, TypeVar

from nlpo_toolkit.immutable_collections import freeze_mapping


class SequenceItem(Protocol):
    group: str
    source_file: str | None
    section: str | None
    chunk_index: int
    sentence_index: int
    token_index: int
    global_token_index: int
    sentence: str
    token: str
    lemma: str | None
    included: bool


TSequenceItem = TypeVar("TSequenceItem", bound=SequenceItem)


@dataclass(frozen=True, order=True)
class TokenSequenceId:
    group: str
    source_file: str | None
    section: str | None
    chunk_index: int
    sentence_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.group, str) or not self.group:
            raise ValueError("Token sequence group must be a non-empty string.")
        if self.chunk_index < 0:
            raise ValueError(f"Negative chunk index {self.chunk_index} for group={self.group!r}.")
        if self.sentence_index < 0:
            raise ValueError(
                f"Negative sentence index {self.sentence_index} for group={self.group!r}, "
                f"file={self.source_file!r}, section={self.section!r}, chunk={self.chunk_index}."
            )


@dataclass(frozen=True)
class TokenSequence(Generic[TSequenceItem]):
    id: TokenSequenceId
    items: tuple[TSequenceItem, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        if not self.items:
            raise ValueError(f"Token sequence must not be empty: {self.id!r}.")
        previous: tuple[int, int] | None = None
        seen_token_indexes: set[int] = set()
        for item in self.items:
            item_id = TokenSequenceId(
                item.group, item.source_file, item.section,
                item.chunk_index, item.sentence_index,
            )
            if item_id != self.id:
                raise ValueError(
                    f"Token sequence ID mismatch: expected={self.id!r}, actual={item_id!r}, "
                    f"token_index={item.token_index}, global_token_index={item.global_token_index}."
                )
            if item.token_index < 0 or item.global_token_index < 0:
                raise ValueError(
                    f"Negative token position in {self.id!r}: token_index={item.token_index}, "
                    f"global_token_index={item.global_token_index}."
                )
            if item.token_index in seen_token_indexes:
                raise ValueError(
                    f"Duplicate token index {item.token_index} in {self.id!r}; "
                    f"global_token_index={item.global_token_index}."
                )
            position = (item.token_index, item.global_token_index)
            if previous is not None and position < previous:
                raise ValueError(f"Token sequence items are not in canonical order: {self.id!r}.")
            seen_token_indexes.add(item.token_index)
            previous = position


@dataclass(frozen=True)
class TokenLocation(Generic[TSequenceItem]):
    sequence: TokenSequence[TSequenceItem]
    offset: int

    def __post_init__(self) -> None:
        if not 0 <= self.offset < len(self.sequence.items):
            raise ValueError(
                f"Token location offset {self.offset} is outside sequence of "
                f"length {len(self.sequence.items)}: {self.sequence.id!r}."
            )

    @property
    def item(self) -> TSequenceItem:
        return self.sequence.items[self.offset]


@dataclass(frozen=True)
class TokenSequenceCollection(Generic[TSequenceItem]):
    sequences: tuple[TokenSequence[TSequenceItem], ...]
    locations_by_global_index: Mapping[int, TokenLocation[TSequenceItem]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequences", tuple(self.sequences))
        object.__setattr__(
            self,
            "locations_by_global_index",
            freeze_mapping(self.locations_by_global_index),
        )

    def require_location(self, global_token_index: int) -> TokenLocation[TSequenceItem]:
        location = self.optional_location(global_token_index)
        if location is None:
            raise ValueError(f"Unknown global token index {global_token_index}.")
        return location

    def optional_location(self, global_token_index: int) -> TokenLocation[TSequenceItem] | None:
        return self.locations_by_global_index.get(global_token_index)
