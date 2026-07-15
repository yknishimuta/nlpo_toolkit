from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitMetadata:
    branch: str | None
    commit: str | None
    dirty: bool


def _read_git_value(project_root: Path, *args: str) -> str | None:
    try:
        process = subprocess.run(
            ["git", *args], cwd=project_root, capture_output=True,
            text=True, check=False,
        )
    except OSError:
        return None
    return process.stdout.strip() or None if process.returncode == 0 else None


def read_git_metadata(project_root: Path) -> GitMetadata:
    return GitMetadata(
        branch=_read_git_value(project_root, "branch", "--show-current"),
        commit=_read_git_value(project_root, "rev-parse", "HEAD"),
        dirty=bool(_read_git_value(project_root, "status", "--porcelain")),
    )
