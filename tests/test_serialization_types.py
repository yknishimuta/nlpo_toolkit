import pytest

from nlpo_toolkit.serialization.types import SerializationTypeError, validate_json_value


def test_nested_json_values_and_scalar_types():
    value = validate_json_value({"items": [True, 1, 1.5, "x", None]})
    assert value == {"items": [True, 1, 1.5, "x", None]}
    assert type(value["items"][0]) is bool
    assert type(value["items"][1]) is int


@pytest.mark.parametrize("raw", [{1: "x"}, {"x": object()}, {"x": float("nan")}])
def test_non_json_values_fail_clearly(raw):
    with pytest.raises(SerializationTypeError):
        validate_json_value(raw)
