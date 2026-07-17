from __future__ import annotations

from pathlib import Path

from nlpo_toolkit.corpus_analysis.config.models import GroupConfig
from nlpo_toolkit.corpus_analysis.corpus import resolve_group_files


def test_group_resolution_without_cleaned_dir_preserves_normal_pattern(tmp_path: Path):
    source = tmp_path / "data" / "a.txt"
    source.parent.mkdir()
    source.write_text("a", encoding="utf-8")
    groups = {"g": GroupConfig(files=("data/*.txt",))}

    result = resolve_group_files(groups=groups, project_root=tmp_path, cleaned_dir=None)

    assert result["g"] == (source.resolve(),)
    assert groups["g"].files == ("data/*.txt",)


def test_group_resolution_expands_cleaned_dir_and_preserves_pattern_order(tmp_path: Path):
    cleaned_dir = tmp_path / "cleaned"
    other_dir = tmp_path / "other"
    cleaned_dir.mkdir()
    other_dir.mkdir()
    cleaned = cleaned_dir / "a.txt"
    other = other_dir / "b.txt"
    cleaned.write_text("a", encoding="utf-8")
    other.write_text("b", encoding="utf-8")
    groups = {"g": GroupConfig(files=("{cleaned_dir}/*.txt", "other/*.txt"))}

    result = resolve_group_files(groups=groups, project_root=tmp_path, cleaned_dir=cleaned_dir)

    assert result["g"] == (cleaned.resolve(), other.resolve())

