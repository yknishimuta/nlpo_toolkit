from __future__ import annotations

from pathlib import Path

from nlpo_toolkit.corpus_analysis.config.models import GroupConfig
from nlpo_toolkit.corpus_analysis.corpus import resolve_group_files


def test_placeholder_expand_then_glob_finds_files(tmp_path: Path):
    cleaned_dir = tmp_path / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    # matched
    f1 = cleaned_dir / "a.txt"
    f2 = cleaned_dir / "b.txt"
    f1.write_text("a", encoding="utf-8")
    f2.write_text("b", encoding="utf-8")

    # not matched (suffix違い)
    (cleaned_dir / "c.md").write_text("c", encoding="utf-8")

    files = resolve_group_files(
        groups={"g": GroupConfig(files=("{cleaned_dir}/*.txt",))},
        project_root=tmp_path,
        cleaned_dir=cleaned_dir,
    )["g"]

    assert files == tuple(sorted([f1.resolve(), f2.resolve()]))


def test_placeholder_expand_then_glob_recursive(tmp_path: Path):
    cleaned_dir = tmp_path / "cleaned"
    (cleaned_dir / "sub").mkdir(parents=True, exist_ok=True)

    # matched recursively
    f1 = cleaned_dir / "sub" / "x.txt"
    f1.write_text("x", encoding="utf-8")

    files = resolve_group_files(
        groups={"g": GroupConfig(files=("{cleaned_dir}/**/*.txt",))},
        project_root=tmp_path,
        cleaned_dir=cleaned_dir,
    )["g"]

    assert files == (f1.resolve(),)
