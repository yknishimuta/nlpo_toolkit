import pytest

from nlpo_toolkit.corpus_analysis.token_sequences.context import build_token_context
from nlpo_toolkit.corpus_analysis.token_sequences.grouping import build_token_sequence_collection


@pytest.mark.parametrize("offset,window,left,right", [
    (0, 0, (), ()), (0, 2, (), (",", "virum")),
    (1, 9, ("arma",), ("virum",)), (2, 2, ("arma", ","), ()),
])
def test_context_windows_use_sequence_tokens(item_type, offset, window, left, right):
    items = [
        item_type(token="arma", token_index=0, global_token_index=0),
        item_type(token=",", token_index=1, global_token_index=1, included=False),
        item_type(token="virum", token_index=2, global_token_index=2),
    ]
    collection = build_token_sequence_collection(items)
    context = build_token_context(collection.require_location(offset), window=window)
    assert (context.left, context.node, context.right) == (left, items[offset].token, right)


def test_context_never_uses_sentence_text_or_other_sequence(item_type):
    collection = build_token_sequence_collection([
        item_type(sentence="invented words around token"),
        item_type(group="h", token="other", global_token_index=1),
    ])
    context = build_token_context(collection.require_location(0), window=10)
    assert context.left == context.right == ()
