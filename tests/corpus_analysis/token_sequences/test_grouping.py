import pytest

from nlpo_toolkit.corpus_analysis.token_sequences.grouping import (
    TokenSequenceError, build_token_sequence_collection,
)


@pytest.mark.parametrize("field,value", [
    ("group", "h"), ("source_file", "b"), ("section", "t"),
    ("chunk_index", 1), ("sentence_index", 1),
])
def test_every_identity_field_splits_sequences(item_type, field, value):
    second = item_type(token="b", token_index=0, global_token_index=1, **{field: value})
    collection = build_token_sequence_collection([item_type(), second])
    assert len(collection.sequences) == 2


def test_grouping_orders_tokens_and_sequences_and_keeps_excluded(item_type):
    items = [
        item_type(token="later", token_index=1, global_token_index=11, included=False),
        item_type(group="h", global_token_index=20),
        item_type(token="first", token_index=0, global_token_index=10),
    ]
    collection = build_token_sequence_collection(items)
    assert [s.id.group for s in collection.sequences] == ["g", "h"]
    assert [i.token for i in collection.sequences[0].items] == ["first", "later"]
    assert collection.require_location(11).item.included is False


def test_empty_and_duplicate_positions(item_type):
    assert build_token_sequence_collection([]).sequences == ()
    with pytest.raises(TokenSequenceError, match="Duplicate token index"):
        build_token_sequence_collection([item_type(), item_type(token="b", global_token_index=1)])
    with pytest.raises(TokenSequenceError, match="Duplicate global"):
        build_token_sequence_collection([item_type(), item_type(group="h")])


@pytest.mark.parametrize("field", ["chunk_index", "sentence_index", "token_index", "global_token_index"])
def test_negative_indexes_are_rejected(item_type, field):
    with pytest.raises(TokenSequenceError, match="Negative"):
        build_token_sequence_collection([item_type(**{field: -1})])
