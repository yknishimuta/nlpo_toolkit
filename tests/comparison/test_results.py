from dataclasses import FrozenInstanceError

import pytest

from nlpo_toolkit.comparison.config import ComparisonSpec
from nlpo_toolkit.comparison.engine import FrequencyTable
from nlpo_toolkit.comparison.results import (
    ConfiguredComparisonResult, PairwiseComparisonResult, PairwiseComparisonRow,
)


def test_configured_result_derives_values_without_copying_rows():
    a = FrequencyTable.from_counts("a", {"x": 2})
    b = FrequencyTable.from_counts("b", {"x": 1})
    row = PairwiseComparisonRow("x", 2, 1, 3, 1, 1, 0, 1, 0, 0, "equal")
    engine_result = PairwiseComparisonResult(a, b, 1, 1, 1, (row,))
    result = ConfiguredComparisonResult(
        ComparisonSpec(name="c", group_a="a", group_b="b"),
        "lemma", engine_result,
    )
    assert result.rows is engine_result.rows
    assert result.group_a_tokens == 2
    assert result.rows_before_filter == result.rows_after_filter == 1
    with pytest.raises(FrozenInstanceError):
        result.analysis_unit = "token"
