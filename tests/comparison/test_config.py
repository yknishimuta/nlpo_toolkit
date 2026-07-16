import pytest
from pydantic import ValidationError

from nlpo_toolkit.comparison.config import ComparisonSortConfig, ComparisonSpec


def test_config_is_strict_stripped_and_has_no_report():
    spec = ComparisonSpec(name=" c ", group_a=" a ", group_b=" b ")
    assert (spec.name, spec.group_a, spec.group_b) == ("c", "a", "b")
    with pytest.raises(ValidationError):
        ComparisonSpec(name="c", group_a="a", group_b="a")
    with pytest.raises(ValidationError):
        ComparisonSpec(name="c", group_a="a", group_b="b", report=True)
    with pytest.raises(ValidationError):
        ComparisonSortConfig(descending=1)


@pytest.mark.parametrize("values", [
    {"scale": 0}, {"scale": 1.5}, {"zero_correction": 0},
    {"min_total_count": 0}, {"sort": {"by": "bad"}},
])
def test_config_rejects_invalid_values(values):
    with pytest.raises(ValidationError):
        ComparisonSpec(name="c", group_a="a", group_b="b", **values)
