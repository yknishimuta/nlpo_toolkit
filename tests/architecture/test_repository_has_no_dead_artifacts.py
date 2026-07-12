from __future__ import annotations

import subprocess
from pathlib import Path


def test_generated_artifacts_are_not_tracked() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    forbidden = [
        path
        for path in tracked
        if (
            ".egg-info/" in path
            or "/__pycache__/" in path
            or path.startswith("build/")
            or path.startswith("dist/")
            or path.endswith((".pyc", ".pyo", "/.DS_Store"))
            or path == ".DS_Store"
        )
    ]

    assert forbidden == []


def test_removed_repository_config_files_do_not_return() -> None:
    removed = (Path("config/exclude_lemmas.txt"),)

    assert [str(path) for path in removed if path.exists()] == []


def test_repository_has_no_removed_config_references() -> None:
    forbidden = {
        "exclude_lemmas",
        "exclude_lemmas_file",
        "vocab_path",
        "lemma_cache",
    }
    paths = (Path("config/groups.config.yml"), Path("README.md"))
    offenders = [
        (str(path), fragment)
        for path in paths
        for fragment in forbidden
        if fragment in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
