import pytest

from nlpo_toolkit.backends.transformers_backend import (
    NLPBackendUnavailableError, parse_transformer_outputs,
)


def test_valid_transformer_output_is_typed():
    result = parse_transformer_outputs([
        {"word": "arma", "entity": "NOUN", "start": 0, "end": 4}
    ])
    assert result[0].word == "arma"
    assert result[0].start == 0


@pytest.mark.parametrize("raw,match", [
    ({"word": "x"}, "sequence"), (["x"], "mapping"),
    ([{"entity": "NOUN"}], "word"),
    ([{"word": "x", "entity": 1}], "entity"),
    ([{"word": "x", "entity": "NOUN", "start": "0"}], "start"),
    ([{"word": "x", "entity": "NOUN", "end": True}], "end"),
])
def test_malformed_transformer_output_is_rejected(raw, match):
    with pytest.raises(NLPBackendUnavailableError, match=match):
        parse_transformer_outputs(raw)
