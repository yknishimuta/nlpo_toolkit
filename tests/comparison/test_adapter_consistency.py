from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest

from nlpo_toolkit.comparison.frequency_io import load_frequency_table
from nlpo_toolkit.comparison import (
    PairwiseComparisonOptions,
    ZeroHandling,
    ZeroHandlingMode,
    compare_pair,
)
from nlpo_toolkit.comparison.configured import ComparisonSpec, compare_counters


def _write_frequency(path: Path, rows: list[tuple[str, int]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lemma", "count"])
        writer.writerows(rows)
    return path


def test_csv_table_and_counter_group_comparison_share_pairwise_values(tmp_path: Path) -> None:
    a_path = _write_frequency(
        tmp_path / "a.csv",
        [("item_common", 10), ("item_a", 8), ("item_rare_a", 1)],
    )
    b_path = _write_frequency(
        tmp_path / "b.csv",
        [("item_common", 20), ("item_b", 7), ("item_rare_b", 1)],
    )

    engine_result = compare_pair(
        load_frequency_table(a_path, label="group_a"),
        load_frequency_table(b_path, label="group_b"),
        options=PairwiseComparisonOptions(
            scale=10000,
            min_total_count=1,
            zero_handling=ZeroHandling(ZeroHandlingMode.ZERO_ONLY, 0.5),
        ),
    )
    group_result = compare_counters(
        counter_a=Counter({"item_common": 10, "item_a": 8, "item_rare_a": 1}),
        counter_b=Counter({"item_common": 20, "item_b": 7, "item_rare_b": 1}),
        spec=ComparisonSpec("comparison_1", "group_a", "group_b"),
        analysis_unit="lemma",
    )

    engine_rows = {row.item: row for row in engine_result.rows}
    group_rows = {row.item: row for row in group_result.rows}

    assert set(engine_rows) == set(group_rows)
    for item, engine_row in engine_rows.items():
        group_row = group_rows[item]
        assert group_row.group_a_count == engine_row.count_a
        assert group_row.group_b_count == engine_row.count_b
        assert group_row.group_a_tokens == engine_result.table_a.total
        assert group_row.group_b_tokens == engine_result.table_b.total
        assert group_row.group_a_rate == pytest.approx(engine_row.rate_a)
        assert group_row.group_b_rate == pytest.approx(engine_row.rate_b)
        assert group_row.rate_difference == pytest.approx(engine_row.rate_difference)
        assert group_row.log_ratio == pytest.approx(engine_row.log_ratio)
        assert group_row.total_count == engine_row.total_count
