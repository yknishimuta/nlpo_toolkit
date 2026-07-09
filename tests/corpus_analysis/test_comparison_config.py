from __future__ import annotations

from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.config import load_config
from nlpo_toolkit.corpus_analysis.comparison import parse_comparison_specs


def _write_cfg(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "groups.config.yml"
    path.write_text(body, encoding="utf-8")
    return path


def _base_yaml(extra: str) -> str:
    return "\n".join(
        [
            "groups:",
            "  corpus_a: {files: [input/corpus_a.txt]}",
            "  corpus_b: {files: [input/corpus_b.txt]}",
            "  corpus_c: {files: [input/corpus_c.txt]}",
            "comparisons:",
            *[f"  {line}" if line else "" for line in extra.splitlines()],
            "",
        ]
    )


def test_load_config_accepts_valid_comparison(tmp_path: Path) -> None:
    cfg_path = _write_cfg(
        tmp_path,
        _base_yaml(
            "\n".join(
                [
                    "- name: comparison_1",
                    "  group_a: corpus_a",
                    "  group_b: corpus_b",
                    "  scale: 10000",
                    "  zero_correction: 0.5",
                    "  min_total_count: 2",
                    "  report: all",
                    "  sort:",
                    "    by: abs_log_ratio",
                    "    descending: false",
                ]
            )
        ),
    )

    specs = parse_comparison_specs(load_config(cfg_path))
    assert len(specs) == 1
    assert specs[0].name == "comparison_1"
    assert specs[0].sort_by == "abs_log_ratio"
    assert specs[0].sort_descending is False


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (
            "- {name: comparison_1, group_a: corpus_a, group_b: corpus_b}\n"
            "- {name: comparison_1, group_a: corpus_a, group_b: corpus_c}",
            "Duplicate comparison name",
        ),
        (
            "- {name: comparison_1, group_a: corpus_a, group_b: corpus_a}",
            "must be different",
        ),
        (
            "- {name: comparison_1, group_a: corpus_a, group_b: missing_group}",
            "unknown group_b",
        ),
        (
            "- {name: comparison_1, group_a: corpus_a, group_b: corpus_b, scale: 0}",
            "scale",
        ),
        (
            "- {name: comparison_1, group_a: corpus_a, group_b: corpus_b, scale: true}",
            "scale",
        ),
        (
            "- {name: comparison_1, group_a: corpus_a, group_b: corpus_b, zero_correction: 0}",
            "zero_correction",
        ),
        (
            "- {name: comparison_1, group_a: corpus_a, group_b: corpus_b, zero_correction: .nan}",
            "zero_correction",
        ),
        (
            "- {name: comparison_1, group_a: corpus_a, group_b: corpus_b, zero_correction: .inf}",
            "zero_correction",
        ),
        (
            "- {name: comparison_1, group_a: corpus_a, group_b: corpus_b, min_total_count: 0}",
            "min_total_count",
        ),
        (
            "- name: comparison_1\n"
            "  group_a: corpus_a\n"
            "  group_b: corpus_b\n"
            "  sort: {by: unsupported}",
            "sort.by",
        ),
    ],
)
def test_load_config_rejects_invalid_comparison_values(
    tmp_path: Path,
    body: str,
    match: str,
) -> None:
    cfg_path = _write_cfg(tmp_path, _base_yaml(body))

    with pytest.raises(ValueError, match=match):
        load_config(cfg_path)


def test_load_config_rejects_non_list_comparisons(tmp_path: Path) -> None:
    cfg_path = _write_cfg(
        tmp_path,
        "\n".join(
            [
                "groups:",
                "  corpus_a: {files: [input/corpus_a.txt]}",
                "comparisons: {}",
                "",
            ]
        ),
    )

    with pytest.raises(ValueError, match="comparisons"):
        load_config(cfg_path)


def test_load_config_rejects_per_file_with_comparisons(tmp_path: Path) -> None:
    cfg_path = _write_cfg(
        tmp_path,
        "\n".join(
            [
                "groups:",
                "  corpus_a: {files: [input/corpus_a.txt]}",
                "  corpus_b: {files: [input/corpus_b.txt]}",
                "grouping:",
                "  mode: per_file",
                "comparisons:",
                "  - name: comparison_1",
                "    group_a: corpus_a",
                "    group_b: corpus_b",
                "",
            ]
        ),
    )

    with pytest.raises(ValueError, match="comparisons cannot be used"):
        load_config(cfg_path)
