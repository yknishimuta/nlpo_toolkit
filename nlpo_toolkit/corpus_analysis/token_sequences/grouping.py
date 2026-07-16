from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .models import (
    SequenceItem, TSequenceItem, TokenLocation, TokenSequence,
    TokenSequenceCollection, TokenSequenceId,
)


class TokenSequenceError(ValueError):
    pass


def token_sequence_id(item: SequenceItem) -> TokenSequenceId:
    try:
        return TokenSequenceId(
            item.group, item.source_file, item.section,
            item.chunk_index, item.sentence_index,
        )
    except ValueError as exc:
        raise TokenSequenceError(str(exc)) from exc


def token_position_key(item: SequenceItem) -> tuple[int, int]:
    if item.token_index < 0 or item.global_token_index < 0:
        raise TokenSequenceError(
            f"Negative token position for {token_sequence_id(item)!r}: "
            f"token_index={item.token_index}, global_token_index={item.global_token_index}."
        )
    return item.token_index, item.global_token_index


def build_token_sequence_collection(
    items: Iterable[TSequenceItem],
) -> TokenSequenceCollection[TSequenceItem]:
    grouped: dict[TokenSequenceId, list[TSequenceItem]] = defaultdict(list)
    global_indexes: set[int] = set()
    for item in items:
        sequence_id = token_sequence_id(item)
        token_position_key(item)
        if item.global_token_index in global_indexes:
            raise TokenSequenceError(
                f"Duplicate global token index {item.global_token_index}: "
                f"{sequence_id!r}, token_index={item.token_index}."
            )
        global_indexes.add(item.global_token_index)
        grouped[sequence_id].append(item)

    sequences: list[TokenSequence[TSequenceItem]] = []
    for sequence_id, sequence_items in grouped.items():
        ordered = tuple(sorted(sequence_items, key=token_position_key))
        try:
            sequences.append(TokenSequence(sequence_id, ordered))
        except ValueError as exc:
            raise TokenSequenceError(str(exc)) from exc
    sequences.sort(key=lambda sequence: (
        min(item.global_token_index for item in sequence.items), sequence.id,
    ))

    locations: dict[int, TokenLocation[TSequenceItem]] = {}
    for sequence in sequences:
        for offset, item in enumerate(sequence.items):
            locations[item.global_token_index] = TokenLocation(sequence, offset)
    return TokenSequenceCollection(tuple(sequences), locations)
