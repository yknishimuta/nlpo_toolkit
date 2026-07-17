from pathlib import Path

import pytest

from nlpo_toolkit.comparison.errors import ComparisonServiceError
from nlpo_toolkit.comparison.results import (
    CsvMultiComparisonResult,
    CsvPairComparisonResult,
    MultiComparisonResult,
    PairwiseComparisonResult,
)
from nlpo_toolkit.comparison.services.csv import CsvComparisonRequest, execute_csv_comparison


def write(path: Path, rows: str) -> Path:
    path.write_text("lemma,count\n" + rows, encoding="utf-8")
    return path


def test_pair_is_typed_sorted_top_and_additive(tmp_path):
    a = write(tmp_path / "a.csv", "x,10\ny,1\n")
    b = write(tmp_path / "b.csv", "x,1\ny,10\n")
    result = execute_csv_comparison(CsvComparisonRequest(
        inputs=(a, b), labels=("a", "b"), sort="term", ascending=True, top=1,
    ))
    assert isinstance(result, CsvPairComparisonResult)
    assert isinstance(result.comparison, PairwiseComparisonResult)
    assert result.comparison.scale == 1
    assert result.rows[0].item == "x"


def test_multi_is_typed_and_default_range_sort(tmp_path):
    paths = tuple(write(tmp_path / f"{label}.csv", rows) for label, rows in (
        ("a", "x,10\ny,1\n"), ("b", "x,1\ny,10\n"), ("c", "x,5\ny,5\n"),
    ))
    result = execute_csv_comparison(CsvComparisonRequest(inputs=paths))
    assert isinstance(result, CsvMultiComparisonResult)
    assert isinstance(result.comparison, MultiComparisonResult)
    assert result.rows[0].range_relative >= result.rows[1].range_relative


@pytest.mark.parametrize("comparison_request,match", [
    (CsvComparisonRequest(inputs=(Path("a"),)), "at least two"),
    (CsvComparisonRequest(inputs=(Path("a"), Path("b")), labels=("x",)), "same length"),
    (CsvComparisonRequest(inputs=(Path("a"), Path("b")), labels=("x", "x")), "unique"),
    (CsvComparisonRequest(inputs=(Path("a"), Path("b")), smoothing=-1), "smoothing"),
    (CsvComparisonRequest(inputs=(Path("a"), Path("b")), top=0), "top"),
])
def test_request_validation_precedes_io(comparison_request, match):
    with pytest.raises(ComparisonServiceError, match=match):
        execute_csv_comparison(comparison_request)
