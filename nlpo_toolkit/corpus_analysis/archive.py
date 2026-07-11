"""Run archive creation from an exact, in-memory run inventory."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .runner_types import ReferencedConfigFile, RunResult

_IGNORED_ARCHIVE_NAMES = {".DS_Store", ".gitkeep"}


class RunArchiveError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArchiveOptions:
    run_name: str | None = None
    runs_dir: Path | str = Path("runs")
    include_cleaned: bool = False
    include_input: bool = False
    command_line: tuple[str, ...] = ()
    created_at: datetime | None = None


@dataclass(frozen=True)
class RunArchiveResult:
    run_dir: Path
    copied_output_count: int
    copied_trace_count: int
    copied_input_count: int
    copied_cleaned_count: int
    copied_config_count: int


@dataclass(frozen=True)
class ArchiveFile:
    source_path: Path
    archive_path: Path
    sha256: str
    size: int


def sanitize_run_name(name: str) -> str:
    raw = str(name).strip()
    if not raw or Path(raw).is_absolute() or "/" in raw or "\\" in raw or ".." in raw:
        raise ValueError("run name must be a safe directory name")
    sanitized = re.sub(r"\s+", "_", raw)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", sanitized):
        raise ValueError("run name must contain only safe characters")
    return sanitized


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, kind: str) -> Path:
    path = Path(path).resolve()
    if not path.exists() or not path.is_file():
        raise RunArchiveError(f"Declared {kind} does not exist: {path}")
    return path


def _unique_dest(root: Path, relative: Path, used: set[Path]) -> Path:
    relative = Path(*[part for part in relative.parts if part not in {"", ".", ".."}])
    candidate = relative
    index = 2
    while candidate in used or (root / candidate).exists():
        candidate = relative.with_name(f"{relative.stem}_{index}{relative.suffix}")
        index += 1
    used.add(candidate)
    return root / candidate


def _copy(files: Iterable[tuple[Path, Path]], root: Path, run_dir: Path) -> list[ArchiveFile]:
    copied: list[ArchiveFile] = []
    used: set[Path] = set()
    for source, relative in files:
        destination = _unique_dest(root, relative, used)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(
            ArchiveFile(source, destination.relative_to(run_dir), file_sha256(destination), destination.stat().st_size)
        )
    return copied


def _source_metadata(files: Iterable[Path]) -> list[dict[str, object]]:
    return [
        {"path": str(path), "sha256": file_sha256(path), "size": path.stat().st_size}
        for path in files
    ]


def _archive_metadata(files: Iterable[ArchiveFile]) -> list[dict[str, object]]:
    return [
        {"source_path": str(item.source_path), "archive_path": str(item.archive_path), "sha256": item.sha256, "size": item.size}
        for item in files
    ]


def _git(project_root: Path, *args: str) -> str | None:
    try:
        process = subprocess.run(["git", *args], cwd=project_root, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return process.stdout.strip() or None if process.returncode == 0 else None


def create_run_archive(*, result: RunResult, options: ArchiveOptions) -> RunArchiveResult:
    project_root = result.plan.project_root.resolve()
    created = options.created_at or datetime.now().astimezone()
    run_name = sanitize_run_name(options.run_name or created.strftime("%Y%m%d-%H%M%S"))
    runs_root = Path(options.runs_dir)
    if not runs_root.is_absolute():
        runs_root = (project_root / runs_root).resolve()
    run_dir = runs_root / run_name
    if run_dir.exists():
        raise RunArchiveError(f"Run archive already exists: {run_dir}")

    outputs = tuple(_require_file(path, "run output") for path in result.output_files)
    traces = tuple(_require_file(path, "run trace") for path in result.trace_files)
    inputs = tuple(_require_file(path, "run input") for path in result.input_files)
    cleaned = tuple(_require_file(path, "cleaned run file") for path in result.cleaned_files)
    snapshots = tuple(item for item in result.config_files if item.copy_to_snapshot)
    externals = tuple(item for item in result.config_files if not item.copy_to_snapshot)
    for item in (*snapshots, *externals):
        _require_file(item.path, f"config reference {item.kind}")

    try:
        run_dir.mkdir(parents=True)
        output_copied = _copy(((path, Path(path.name)) for path in outputs), run_dir / "outputs", run_dir)
        trace_copied = _copy(((path, Path(path.name)) for path in traces), run_dir / "trace", run_dir)
        config_copied = _copy(
            ((_require_file(item.path, "config snapshot"), item.snapshot_path or Path(item.path.name)) for item in snapshots),
            run_dir / "config_snapshot", run_dir,
        )
        input_copied = _copy(
            ((path, path.relative_to(project_root) if path.is_relative_to(project_root) else Path(path.name)) for path in inputs),
            run_dir / "input", run_dir,
        ) if options.include_input else []
        cleaned_root = result.plan.cleaned_dir.resolve() if result.plan.cleaned_dir else project_root
        cleaned_copied = _copy(
            ((path, path.relative_to(cleaned_root) if path.is_relative_to(cleaned_root) else Path(path.name)) for path in cleaned),
            run_dir / "cleaned", run_dir,
        ) if options.include_cleaned else []
        external_metadata = [
            {"kind": item.kind, "path": str(item.path.resolve()), "exists": True, "sha256": file_sha256(item.path), "size": item.path.stat().st_size}
            for item in externals
        ]
        manifest = {
            "run_name": run_name,
            "created_at": created.isoformat(),
            "command_line": list(options.command_line or tuple(sys.argv)),
            "project_root": str(project_root),
            "config_path": str(result.plan.config_path.resolve()),
            "output_dir": str(result.plan.out_dir.resolve()),
            "git": {"branch": _git(project_root, "branch", "--show-current"), "commit": _git(project_root, "rev-parse", "HEAD"), "dirty": bool(_git(project_root, "status", "--porcelain"))},
            "groups_files": {name: [str(path) for path in files] for name, files in result.groups_files.items()},
            "input_files": _source_metadata(inputs),
            "cleaned_files": _source_metadata(cleaned),
            "generated_outputs": _source_metadata(result.generated_outputs),
            "output_files": _archive_metadata(output_copied),
            "copied_outputs": _archive_metadata(output_copied),
            "trace_files": _archive_metadata(trace_copied),
            "config_snapshot_files": _archive_metadata(config_copied),
            "external_references": external_metadata,
            "included_input_files": _archive_metadata(input_copied),
            "included_cleaned_files": _archive_metadata(cleaned_copied),
            "copied_input_files": _archive_metadata(input_copied),
            "copied_cleaned_files": _archive_metadata(cleaned_copied),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (run_dir / "README.md").write_text(
            "# Run Archive\n\n"
            f"- input files: {len(inputs)}\n"
            f"- Included input files: {len(input_copied)}\n"
            f"- Included cleaned files: {len(cleaned_copied)}\n",
            encoding="utf-8",
        )
    except Exception as exc:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        if isinstance(exc, RunArchiveError):
            raise
        raise RunArchiveError(f"Failed to create run archive: {exc}") from exc
    return RunArchiveResult(run_dir, len(output_copied), len(trace_copied), len(input_copied), len(cleaned_copied), len(config_copied))
