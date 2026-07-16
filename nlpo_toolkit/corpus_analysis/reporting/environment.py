from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .models import RuntimeEnvironmentReport


def _command(command: tuple[str, ...], *, cwd: Path) -> str | None:
    try:
        output = subprocess.check_output(command, cwd=cwd, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    return output.decode("utf-8", errors="replace").strip()


def collect_runtime_environment(project_root: Path) -> RuntimeEnvironmentReport:
    root = project_root.resolve()
    return RuntimeEnvironmentReport(
        python_version=sys.version,
        platform=sys.platform,
        executable=Path(sys.executable).resolve(),
        project_root=root,
        git_commit=_command(("git", "rev-parse", "HEAD"), cwd=root),
        git_status=_command(("git", "status", "--porcelain"), cwd=root),
    )
