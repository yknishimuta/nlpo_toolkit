from dataclasses import FrozenInstanceError

import pytest

from nlpo_toolkit.corpus_analysis.token_sequences.models import (
    TokenLocation, TokenSequence, TokenSequenceCollection, TokenSequenceId,
)


def test_sequence_id_is_frozen_hashable_and_preserves_identity():
    value = TokenSequenceId("G", "A", "", 1, 2)
    assert {value} == {TokenSequenceId("G", "A", "", 1, 2)}
    assert value != TokenSequenceId("G", "A", None, 1, 2)
    with pytest.raises(FrozenInstanceError):
        value.group = "x"


def test_sequence_validates_tuple_empty_identity_and_location(item_type):
    identity = TokenSequenceId("g", "a.txt", "s", 0, 0)
    assert isinstance(TokenSequence(identity, [item_type()]).items, tuple)
    with pytest.raises(ValueError, match="empty"):
        TokenSequence(identity, ())
    with pytest.raises(ValueError, match="mismatch"):
        TokenSequence(identity, (item_type(group="x"),))
    sequence = TokenSequence(identity, (item_type(),))
    assert TokenLocation(sequence, 0).item.token == "arma"
    with pytest.raises(ValueError, match="outside"):
        TokenLocation(sequence, 1)


def test_collection_lookup_is_read_only(item_type):
    sequence = TokenSequence(TokenSequenceId("g", "a.txt", "s", 0, 0), (item_type(),))
    location = TokenLocation(sequence, 0)
    collection = TokenSequenceCollection((sequence,), {0: location})
    assert collection.require_location(0) is location
    assert collection.optional_location(9) is None
    with pytest.raises(TypeError):
        collection.locations_by_global_index[1] = location
