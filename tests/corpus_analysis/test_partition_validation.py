from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.config import load_config
from nlpo_toolkit.corpus_analysis.partition_validation import (
    PartitionSpec,
    parse_partition_specs,
    validate_partition,
)


def _spec(*, report: str = "mismatches", parts: tuple[str, ...] = ("part_a", "part_b")):
    return PartitionSpec("split", "whole", parts, "warn", report)


def test_validate_partition_exact_match() -> None:
    result = validate_partition(
        _spec(),
        {
            "whole": Counter({"a": 3, "b": 2}),
            "part_a": Counter({"a": 1, "b": 2}),
            "part_b": Counter({"a": 2}),
        },
    )

    assert result.exact_match is True
    assert result.whole_target_tokens == 5
    assert result.parts_target_tokens == 5
    assert result.mismatched_items == 0
    assert result.matched_items == 2
    assert result.mismatches == []


def test_validate_partition_whole_side_is_larger() -> None:
    result = validate_partition(
        PartitionSpec("split", "whole", ("part_a", "part_b"), "warn", "mismatches"),
        {
            "whole": Counter({"a": 3}),
            "part_a": Counter({"a": 2}),
            "part_b": Counter(),
        },
    )

    assert result.exact_match is False
    assert result.mismatches[0].delta == 1
    assert result.mismatches[0].status == "missing_from_parts"


def test_validate_partition_parts_side_is_larger() -> None:
    result = validate_partition(
        PartitionSpec("split", "whole", ("part_a", "part_b"), "warn", "mismatches"),
        {
            "whole": Counter({"a": 2}),
            "part_a": Counter({"a": 3}),
            "part_b": Counter(),
        },
    )

    assert result.mismatches[0].delta == -1
    assert result.mismatches[0].status == "excess_in_parts"


def test_validate_partition_supports_three_or_more_parts() -> None:
    result = validate_partition(
        _spec(parts=("part_a", "part_b", "part_c")),
        {
            "whole": Counter({"a": 3}),
            "part_a": Counter({"a": 1}),
            "part_b": Counter({"a": 1}),
            "part_c": Counter({"a": 1}),
        },
    )

    assert result.exact_match is True


def test_validate_partition_includes_words_only_in_parts_or_whole() -> None:
    result = validate_partition(
        _spec(report="all"),
        {
            "whole": Counter({"whole_only": 1, "same": 1}),
            "part_a": Counter({"parts_only": 1, "same": 1}),
            "part_b": Counter(),
        },
    )

    rows = {row.item: row for row in result.mismatches}
    assert rows["whole_only"].status == "missing_from_parts"
    assert rows["parts_only"].status == "excess_in_parts"
    assert rows["same"].status == "match"


def test_validate_partition_type_counts_use_parts_union_not_sum() -> None:
    result = validate_partition(
        _spec(),
        {
            "whole": Counter({"a": 2, "b": 1}),
            "part_a": Counter({"a": 1, "b": 1}),
            "part_b": Counter({"a": 1}),
        },
    )

    assert result.whole_types == 2
    assert result.parts_union_types == 2


def test_validate_partition_report_modes() -> None:
    counters = {
        "whole": Counter({"a": 2, "b": 1}),
        "part_a": Counter({"a": 1, "b": 1}),
        "part_b": Counter(),
    }

    mismatch_only = validate_partition(_spec(report="mismatches"), counters)
    all_rows = validate_partition(_spec(report="all"), counters)

    assert [row.item for row in mismatch_only.mismatches] == ["a"]
    assert {row.item for row in all_rows.mismatches} == {"a", "b"}


def _write_cfg(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "groups.config.yml"
    path.write_text(body, encoding="utf-8")
    return path


def test_parse_partition_specs_rejects_unknown_group() -> None:
    with pytest.raises(ValueError, match="unknown group"):
        parse_partition_specs(
            {
                "groups": {"whole": {"files": []}, "part_a": {"files": []}},
                "validations": {
                    "partitions": [
                        {"name": "split", "whole": "whole", "parts": ["part_a", "part_b"]}
                    ]
                },
            }
        )


def test_load_config_rejects_non_mapping_validations(tmp_path: Path) -> None:
    cfg = _write_cfg(
        tmp_path,
        "groups:\n  full: {files: [a.txt]}\nvalidations: []\n",
    )

    with pytest.raises(ValueError, match="validations"):
        load_config(cfg)


def test_load_config_rejects_non_list_partitions(tmp_path: Path) -> None:
    cfg = _write_cfg(
        tmp_path,
        "groups:\n  full: {files: [a.txt]}\nvalidations:\n  partitions: {}\n",
    )

    with pytest.raises(ValueError, match="validations.partitions"):
        load_config(cfg)


def test_load_config_rejects_non_mapping_partition_item(tmp_path: Path) -> None:
    cfg = _write_cfg(
        tmp_path,
        "groups:\n  full: {files: [a.txt]}\nvalidations:\n  partitions: [bad]\n",
    )

    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(cfg)


def test_load_config_rejects_whole_in_parts(tmp_path: Path) -> None:
    cfg = _write_cfg(
        tmp_path,
        "groups:\n  full: {files: [a.txt]}\n  part: {files: [b.txt]}\n"
        "validations:\n  partitions:\n    - name: split\n      whole: full\n"
        "      parts: [full, part]\n",
    )

    with pytest.raises(ValueError, match="whole"):
        load_config(cfg)


def test_load_config_rejects_duplicate_parts(tmp_path: Path) -> None:
    cfg = _write_cfg(
        tmp_path,
        "groups:\n  full: {files: [a.txt]}\n  part: {files: [b.txt]}\n"
        "validations:\n  partitions:\n    - name: split\n      whole: full\n"
        "      parts: [part, part]\n",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_config(cfg)


def test_load_config_rejects_duplicate_partition_names(tmp_path: Path) -> None:
    cfg = _write_cfg(
        tmp_path,
        "groups:\n  full: {files: [a.txt]}\n  a: {files: [a.txt]}\n  b: {files: [b.txt]}\n"
        "validations:\n  partitions:\n"
        "    - {name: split, whole: full, parts: [a, b]}\n"
        "    - {name: split, whole: full, parts: [a, b]}\n",
    )

    with pytest.raises(ValueError, match="Duplicate partition name"):
        load_config(cfg)


def test_load_config_rejects_invalid_on_mismatch(tmp_path: Path) -> None:
    cfg = _write_cfg(
        tmp_path,
        "groups:\n  full: {files: [a.txt]}\n  a: {files: [a.txt]}\n  b: {files: [b.txt]}\n"
        "validations:\n  partitions:\n"
        "    - {name: split, whole: full, parts: [a, b], on_mismatch: fail}\n",
    )

    with pytest.raises(ValueError, match="on_mismatch"):
        load_config(cfg)


def test_load_config_rejects_invalid_report(tmp_path: Path) -> None:
    cfg = _write_cfg(
        tmp_path,
        "groups:\n  full: {files: [a.txt]}\n  a: {files: [a.txt]}\n  b: {files: [b.txt]}\n"
        "validations:\n  partitions:\n"
        "    - {name: split, whole: full, parts: [a, b], report: summary}\n",
    )

    with pytest.raises(ValueError, match="report"):
        load_config(cfg)


def test_load_config_rejects_grouping_per_file_with_partitions(tmp_path: Path) -> None:
    cfg = _write_cfg(
        tmp_path,
        "groups:\n  full: {files: [a.txt]}\n  a: {files: [a.txt]}\n  b: {files: [b.txt]}\n"
        "grouping:\n  mode: per_file\n"
        "validations:\n  partitions:\n"
        "    - {name: split, whole: full, parts: [a, b]}\n",
    )

    with pytest.raises(ValueError, match="per_file"):
        load_config(cfg)
