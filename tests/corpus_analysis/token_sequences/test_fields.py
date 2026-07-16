import pytest

from nlpo_toolkit.corpus_analysis.token_sequences.fields import token_field_value


def test_fields_are_raw_and_none_lemma_is_empty(item_type):
    item = item_type(token=" ArMa ", lemma=" LEMMA ")
    assert token_field_value(item, "token") == " ArMa "
    assert token_field_value(item, "lemma") == " LEMMA "
    assert token_field_value(item_type(lemma=None), "lemma") == ""
    with pytest.raises(ValueError):
        token_field_value(item, "bad")
