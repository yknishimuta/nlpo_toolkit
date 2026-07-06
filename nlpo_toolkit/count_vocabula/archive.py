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
from typing import Any, Iterable

import yaml

from .io_utils import expand_globs
from .preprocess import expand_cleaned_dir_placeholders, resolve_cleaner_output_dir
from .runner import _resolve_project_path


class RunArchiveError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArchiveFile:
    source_path: Path
    archive_path: Path
    sha256: str
    size: int


_RUN_NAME_SAFE_RE = re.compile(r"[^0-9A-Za-z._-]+")
_UNDERSCORE_RE = re.compile(r"_+")
_IGNORED_ARCHIVE_NAMES = {".DS_Store", ".gitkeep"}


def sanitize_run_name(name: str) -> str:
    raw = str(name).strip()
    if not raw:
        raise ValueError("run name must not be empty")
    raw_path = Path(raw)
    if raw_path.is_absolute() or "/" in raw or "\\" in raw or ".." in raw_path.parts:
        raise ValueError("run name must be a single safe directory name")

    sanitized = _RUN_NAME_SAFE_RE.sub("_", raw)
    sanitized = _UNDERSCORE_RE.sub("_", sanitized).strip("._-")
    if not sanitized:
        raise ValueError("run name must contain at least one alphanumeric character")
    if sanitized in {".", ".."}:
        raise ValueError("run name must be a safe directory name")
    return sanitized


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_path(base: Path, raw: Any) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise RunArchiveError(f"YAML root must be a mapping: {path}")
    return data


def _append_existing(paths: list[Path], path: Path | None) -> None:
    if path is not None and path.exists() and path.is_file() and path not in paths:
        paths.append(path)


def collect_referenced_config_files(
    config: dict[str, Any],
    project_root: Path,
    config_path: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    paths: list[Path] = [config_path]
    external_refs: list[dict[str, Any]] = []

    preprocess = config.get("preprocess") or {}
    cleaner_config_path: Path | None = None
    if isinstance(preprocess, dict) and preprocess.get("kind") == "cleaner":
        cleaner_raw = preprocess.get("config")
        if cleaner_raw:
            cleaner_config_path = _resolve_project_path(project_root, cleaner_raw)
            _append_existing(paths, cleaner_config_path)
            if cleaner_config_path.exists():
                cleaner_cfg = _load_yaml(cleaner_config_path)
                for key in ("rules_path", "lexicon_map_path"):
                    raw = cleaner_cfg.get(key)
                    if raw:
                        _append_existing(paths, _resolve_path(cleaner_config_path.parent, raw))

    dictcheck = config.get("dictcheck") or {}
    if isinstance(dictcheck, dict):
        lemma_normalize = dictcheck.get("lemma_normalize")
        if lemma_normalize:
            _append_existing(paths, _resolve_project_path(project_root, lemma_normalize))

        wordlist = dictcheck.get("wordlist")
        if wordlist:
            wordlist_path = _resolve_project_path(project_root, wordlist)
            ref: dict[str, Any] = {
                "kind": "dictcheck.wordlist",
                "path": str(wordlist_path),
                "exists": wordlist_path.exists(),
            }
            if wordlist_path.exists() and wordlist_path.is_file():
                ref["sha256"] = file_sha256(wordlist_path)
                ref["size"] = wordlist_path.stat().st_size
            external_refs.append(ref)

    ref_tags = config.get("ref_tags") or {}
    if isinstance(ref_tags, dict):
        ref_file = ref_tags.get("patterns") or ref_tags.get("ref_tags_file")
        if ref_file:
            _append_existing(paths, _resolve_project_path(project_root, ref_file))

    filters = config.get("filter") or config.get("filters") or {}
    if isinstance(filters, dict):
        roman = filters.get("roman_exceptions_file") or filters.get("roman_exception_files")
        if roman:
            _append_existing(paths, _resolve_project_path(project_root, roman))
        exclude = filters.get("exclude_lemmas") or filters.get("exclude_lemmas_file")
        if exclude:
            _append_existing(paths, _resolve_project_path(project_root, exclude))

    return paths, external_refs


def _safe_relative(path: Path, root: Path) -> Path:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = Path("external") / path.name
    if rel.name in {"", ".", ".."}:
        rel = Path(path.name)
    return rel


def _unique_dest(dest_root: Path, rel_path: Path, used: set[Path]) -> Path:
    rel_path = Path(*[p for p in rel_path.parts if p not in {"", ".", ".."}])
    candidate = rel_path
    i = 2
    while candidate in used or (dest_root / candidate).exists():
        candidate = rel_path.with_name(f"{rel_path.stem}_{i}{rel_path.suffix}")
        i += 1
    used.add(candidate)
    return dest_root / candidate


def _copy_file(src: Path, dest: Path, run_dir: Path) -> ArchiveFile:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return ArchiveFile(
        source_path=src.resolve(),
        archive_path=dest.relative_to(run_dir),
        sha256=file_sha256(dest),
        size=dest.stat().st_size,
    )


def _copy_flat(files: Iterable[Path], dest_root: Path, run_dir: Path) -> list[ArchiveFile]:
    copied: list[ArchiveFile] = []
    used: set[Path] = set()
    for src in files:
        if not _is_archivable_file(src):
            continue
        dest = _unique_dest(dest_root, Path(src.name), used)
        copied.append(_copy_file(src, dest, run_dir))
    return copied


def _copy_preserve_root(
    files: Iterable[Path],
    *,
    source_root: Path,
    dest_root: Path,
    run_dir: Path,
) -> list[ArchiveFile]:
    copied: list[ArchiveFile] = []
    used: set[Path] = set()
    for src in files:
        if not _is_archivable_file(src):
            continue
        dest = _unique_dest(dest_root, _safe_relative(src, source_root), used)
        copied.append(_copy_file(src, dest, run_dir))
    return copied


def _metadata(files: Iterable[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in files:
        if path.exists() and path.is_file():
            out.append(
                {
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                    "size": path.stat().st_size,
                }
            )
    return out


def _source_metadata(files: Iterable[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }
        for path in files
        if _is_archivable_file(path)
    ]


def _is_archivable_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.name not in _IGNORED_ARCHIVE_NAMES


def _archive_metadata(files: Iterable[ArchiveFile]) -> list[dict[str, Any]]:
    return [
        {
            "source_path": str(f.source_path),
            "archive_path": str(f.archive_path),
            "sha256": f.sha256,
            "size": f.size,
        }
        for f in files
    ]


def _collect_output_files(out_dir: Path) -> list[Path]:
    run_meta_path = out_dir / "run_meta.json"
    if run_meta_path.exists():
        try:
            meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        generated = meta.get("generated_outputs")
        if isinstance(generated, list):
            files = [Path(str(p)).resolve() for p in generated if str(p).strip()]
            return list(dict.fromkeys(p for p in files if _is_archivable_file(p)))

    if not out_dir.exists():
        return []
    files: list[Path] = []
    for pattern in ("noun_frequency*.csv", "ref_tags_*.csv"):
        files.extend(sorted(p for p in out_dir.glob(pattern) if _is_archivable_file(p)))
    for name in ("summary.txt", "run_meta.json"):
        p = out_dir / name
        if _is_archivable_file(p):
            files.append(p)
    return list(dict.fromkeys(files))


def _trace_base_path(config: dict[str, Any], project_root: Path, out_dir: Path) -> Path | None:
    trace_cfg = config.get("trace") or {}
    if not isinstance(trace_cfg, dict) or not bool(trace_cfg.get("enabled", False)):
        return None
    raw_path = trace_cfg.get("path")
    if raw_path:
        return _resolve_project_path(project_root, raw_path)
    return out_dir / "trace.tsv"


def _collect_trace_files(config: dict[str, Any], project_root: Path, out_dir: Path) -> list[Path]:
    base = _trace_base_path(config, project_root, out_dir)
    if base is None:
        return []
    files: list[Path] = []
    if _is_archivable_file(base):
        files.append(base)
    if base.parent.exists():
        pattern = f"{base.stem}_*{base.suffix or '.tsv'}"
        files.extend(sorted(p for p in base.parent.glob(pattern) if _is_archivable_file(p)))
    return list(dict.fromkeys(files))


def _cleaned_dir(config: dict[str, Any], project_root: Path) -> Path | None:
    preprocess = config.get("preprocess") or {}
    if not isinstance(preprocess, dict) or preprocess.get("kind") != "cleaner":
        return None
    cleaner_raw = preprocess.get("config")
    if not cleaner_raw:
        return None
    cleaner_path = _resolve_project_path(project_root, cleaner_raw)
    if not cleaner_path.exists():
        return None
    return resolve_cleaner_output_dir(cleaner_path)


def _collect_group_files(
    config: dict[str, Any],
    project_root: Path,
    cleaned_dir: Path | None,
) -> list[Path]:
    groups = config.get("groups") or {}
    if not isinstance(groups, dict):
        return []
    files: list[Path] = []
    for group_def in groups.values():
        if not isinstance(group_def, dict):
            continue
        patterns = group_def.get("files") or []
        if not isinstance(patterns, list):
            continue
        resolved_patterns = [
            str(_resolve_project_path(project_root, p))
            if not Path(str(p)).is_absolute() and "{cleaned_dir}" not in str(p)
            else str(p)
            for p in patterns
        ]
        resolved_patterns = expand_cleaned_dir_placeholders(resolved_patterns, cleaned_dir)
        files.extend(expand_globs(resolved_patterns))
    return list(dict.fromkeys(p.resolve() for p in files if _is_archivable_file(p)))


def _cleaner_config_path(config: dict[str, Any], project_root: Path) -> Path | None:
    preprocess = config.get("preprocess") or {}
    if not isinstance(preprocess, dict) or preprocess.get("kind") != "cleaner":
        return None
    cleaner_raw = preprocess.get("config")
    if not cleaner_raw:
        return None
    cleaner_path = _resolve_project_path(project_root, cleaner_raw)
    if not cleaner_path.exists():
        return None
    return cleaner_path


def _collect_cleaner_input_files(config: dict[str, Any], project_root: Path) -> list[Path]:
    cleaner_path = _cleaner_config_path(config, project_root)
    if cleaner_path is None:
        return []

    cleaner_cfg = _load_yaml(cleaner_path)
    raw_input = cleaner_cfg.get("input")
    if not raw_input:
        return []

    input_path = _resolve_path(cleaner_path.parent, raw_input)
    if input_path.is_file():
        return [input_path.resolve()] if _is_archivable_file(input_path) else []
    if input_path.is_dir():
        return sorted(
            p.resolve()
            for p in input_path.iterdir()
            if _is_archivable_file(p) and p.suffix.lower() == ".txt"
        )
    return []


def _collect_input_files(
    config: dict[str, Any],
    project_root: Path,
    group_files: list[Path],
) -> list[Path]:
    cleaner_input_files = _collect_cleaner_input_files(config, project_root)
    if cleaner_input_files:
        return cleaner_input_files
    return group_files


def _collect_cleaned_files(cleaned_dir: Path | None, group_files: list[Path]) -> list[Path]:
    if cleaned_dir is None:
        return []
    cleaned_root = cleaned_dir.resolve()
    cleaned_files: list[Path] = []
    for path in group_files:
        try:
            path.resolve().relative_to(cleaned_root)
        except ValueError:
            continue
        if _is_archivable_file(path):
            cleaned_files.append(path.resolve())
    return list(dict.fromkeys(cleaned_files))


def _git_value(project_root: Path, args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _git_dirty(project_root: Path) -> bool | None:
    status = _git_value(project_root, ["status", "--porcelain"])
    if status is None:
        return None
    return bool(status)


def _format_command_line(command_line: list[str] | None) -> str:
    if not command_line:
        return ""
    return " ".join(command_line)


def _write_readme(
    *,
    run_dir: Path,
    command_line: list[str] | None,
    input_count: int,
    copied_input_count: int,
    copied_cleaned_count: int,
    output_files: list[ArchiveFile],
    project_root: Path,
    config_path: Path,
) -> None:
    lines = [
        "# Run Archive",
        "",
        "## Command",
        "",
        "```bash",
        _format_command_line(command_line),
        "```",
        "",
        "## Inputs",
        "",
        f"- input files: {input_count}",
        f"- Included input files: {copied_input_count}",
        f"- Included cleaned files: {copied_cleaned_count}",
        "",
        "## Outputs",
        "",
    ]
    if output_files:
        lines.extend(f"- {f.archive_path}" for f in output_files)
    else:
        lines.append("- (none)")
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            f"- project_root: {project_root}",
            f"- config_path: {config_path}",
            "- config snapshot: config_snapshot/",
            "- manifest: manifest.json",
            "",
        ]
    )
    (run_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def create_run_archive(
    *,
    project_root: Path,
    config_path: Path,
    config: dict[str, Any],
    run_name: str | None = None,
    runs_dir: Path | str = "runs",
    include_cleaned: bool = False,
    include_input: bool = False,
    command_line: list[str] | None = None,
    created_at: datetime | None = None,
) -> Path:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    created = created_at or datetime.now().astimezone()
    safe_name = sanitize_run_name(run_name or created.strftime("%Y%m%d-%H%M%S"))

    runs_root = Path(runs_dir)
    if not runs_root.is_absolute():
        runs_root = (project_root / runs_root).resolve()
    run_dir = runs_root / safe_name
    if run_dir.exists():
        raise RunArchiveError(f"Run archive already exists: {run_dir}")

    out_dir = Path(config.get("out_dir", "output"))
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()

    cleaned_dir = _cleaned_dir(config, project_root)
    group_files = _collect_group_files(config, project_root, cleaned_dir)
    input_files = _collect_input_files(config, project_root, group_files)
    cleaned_files = _collect_cleaned_files(cleaned_dir, group_files)
    output_sources = _collect_output_files(out_dir)
    trace_sources = _collect_trace_files(config, project_root, out_dir)
    config_sources, external_refs = collect_referenced_config_files(
        config,
        project_root,
        config_path,
    )

    try:
        run_dir.mkdir(parents=True)
        output_copied = _copy_flat(output_sources, run_dir / "outputs", run_dir)
        trace_copied = _copy_flat(trace_sources, run_dir / "trace", run_dir)
        config_copied = _copy_preserve_root(
            config_sources,
            source_root=project_root,
            dest_root=run_dir / "config_snapshot",
            run_dir=run_dir,
        )
        cleaned_copied: list[ArchiveFile] = []
        if include_cleaned:
            cleaned_copied = _copy_preserve_root(
                cleaned_files,
                source_root=cleaned_dir or project_root,
                dest_root=run_dir / "cleaned",
                run_dir=run_dir,
            )
        input_copied: list[ArchiveFile] = []
        if include_input:
            input_copied = _copy_preserve_root(
                input_files,
                source_root=project_root,
                dest_root=run_dir / "input",
                run_dir=run_dir,
            )

        trace_base = _trace_base_path(config, project_root, out_dir)
        manifest = {
            "run_name": safe_name,
            "created_at": created.isoformat(),
            "command_line": command_line or sys.argv,
            "project_root": str(project_root),
            "config_path": str(config_path),
            "output_dir": str(out_dir),
            "trace_path": str(trace_base) if trace_base else None,
            "git": {
                "branch": _git_value(project_root, ["branch", "--show-current"]),
                "commit": _git_value(project_root, ["rev-parse", "HEAD"]),
                "dirty": _git_dirty(project_root),
            },
            "input_files": _metadata(input_files),
            "cleaned_files": _metadata(cleaned_files),
            "generated_outputs": _source_metadata(output_sources),
            "output_files": _archive_metadata(output_copied),
            "copied_outputs": _archive_metadata(output_copied),
            "trace_files": _archive_metadata(trace_copied),
            "config_snapshot_files": _archive_metadata(config_copied),
            "included_cleaned_files": _archive_metadata(cleaned_copied),
            "included_input_files": _archive_metadata(input_copied),
            "copied_cleaned_files": _archive_metadata(cleaned_copied),
            "copied_input_files": _archive_metadata(input_copied),
            "external_references": external_refs,
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_readme(
            run_dir=run_dir,
            command_line=command_line or sys.argv,
            input_count=len(input_files),
            copied_input_count=len(input_copied),
            copied_cleaned_count=len(cleaned_copied),
            output_files=output_copied,
            project_root=project_root,
            config_path=config_path,
        )
    except Exception as exc:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        if isinstance(exc, RunArchiveError):
            raise
        raise RunArchiveError(f"Failed to create run archive: {exc}") from exc

    return run_dir
