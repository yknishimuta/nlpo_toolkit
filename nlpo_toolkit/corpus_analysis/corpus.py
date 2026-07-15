from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from .config import AppConfig, GroupConfig, GroupingMode
from .config_references import ResolvedConfigFiles
from .corpus_errors import CorpusPreparationError
from .io_utils import expand_globs, read_concat
from .normalizer import normalize_text
from .preprocess import expand_cleaned_dir_placeholders
from .ref_tags import RefTagPattern, load_ref_tag_patterns, strip_and_count_ref_tags


_LABEL_SAFE_RE = re.compile(r"[^0-9A-Za-z]+")
_IGNORED_CLEANED_NAMES = {".DS_Store", ".gitkeep"}


@dataclass(frozen=True)
class CorpusWorkItem:
    label: str
    files: tuple[Path, ...]


@dataclass(frozen=True)
class PreparedCorpus:
    label: str
    files: tuple[Path, ...]
    raw_text: str
    prepared_text: str
    ref_tag_counts: Counter[str]


@dataclass(frozen=True)
class ResolvedCorpora:
    cleaned_dir: Path | None
    group_files: Mapping[str, tuple[Path, ...]]
    work_items: tuple[CorpusWorkItem, ...]
    mode: GroupingMode


def resolve_project_path(project_root: Path, raw: object) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def sanitize_label(value: str) -> str:
    label = _LABEL_SAFE_RE.sub("_", str(value)).strip("_").lower()
    return label or "file"


def label_from_file(path: Path) -> str:
    return sanitize_label(path.stem)


def _unique_label(base: str, used: set[str]) -> str:
    label = base
    i = 2
    while label in used:
        label = f"{base}_{i}"
        i += 1
    used.add(label)
    return label


def resolve_group_files(
    *,
    groups: Mapping[str, GroupConfig],
    project_root: Path,
    cleaned_dir: Path | None,
) -> Mapping[str, tuple[Path, ...]]:
    resolved: dict[str, tuple[Path, ...]] = {}
    for group_name, group_def in groups.items():
        patterns = [
            str(resolve_project_path(project_root, p))
            if not Path(str(p)).is_absolute() and "{cleaned_dir}" not in str(p)
            else str(p)
            for p in group_def.files
        ]
        patterns = expand_cleaned_dir_placeholders(patterns, cleaned_dir)
        resolved[group_name] = tuple(expand_globs(patterns))
    return MappingProxyType(resolved)


def cleaned_txt_files(cleaned_dir: Path | None) -> tuple[Path, ...]:
    if cleaned_dir is None:
        raise CorpusPreparationError("--auto-single-cleaned was enabled, but cleaned_dir is not available")
    cleaned_dir = Path(cleaned_dir).resolve()
    if not cleaned_dir.exists():
        raise CorpusPreparationError(
            f"--auto-single-cleaned was enabled, but cleaned directory does not exist: {cleaned_dir}"
        )
    return tuple(
        sorted(
            p.resolve()
            for p in cleaned_dir.glob("*.txt")
            if p.is_file() and p.name not in _IGNORED_CLEANED_NAMES
        )
    )


def resolve_auto_single_cleaned_group(
    *,
    cleaned_dir: Path | None,
    group_name: str,
) -> Mapping[str, tuple[Path, ...]]:
    files = cleaned_txt_files(cleaned_dir)
    cleaned_display = str(Path(cleaned_dir).resolve()) if cleaned_dir is not None else "cleaned/"
    if not files:
        raise CorpusPreparationError(
            f"--auto-single-cleaned was enabled, but no .txt files were found in {cleaned_display}"
        )
    if len(files) > 1:
        listed = "\n".join(f"  {p}" for p in files)
        raise CorpusPreparationError(
            "--auto-single-cleaned expected exactly one cleaned .txt file, "
            f"but found {len(files)}:\n{listed}\n\n"
            "Remove stale cleaned files, or specify groups.files explicitly."
        )
    return MappingProxyType({group_name: files})


def build_corpus_work_items(
    *,
    group_files: Mapping[str, tuple[Path, ...]],
    group_by_file: bool,
) -> tuple[CorpusWorkItem, ...]:
    if not group_by_file:
        return tuple(CorpusWorkItem(label=name, files=tuple(files)) for name, files in group_files.items())

    items: list[CorpusWorkItem] = []
    seen_files: set[Path] = set()
    used_labels: set[str] = set()
    for files in group_files.values():
        for file_path in files:
            file_path = file_path.resolve()
            if file_path in seen_files:
                continue
            seen_files.add(file_path)
            label = _unique_label(label_from_file(file_path), used_labels)
            items.append(CorpusWorkItem(label=label, files=(file_path,)))
    return tuple(items)


def resolve_corpus_work_items(
    *,
    config: AppConfig,
    project_root: Path,
    cleaned_dir: Path | None,
    grouping_mode: GroupingMode,
    error_on_empty_group: bool = False,
) -> ResolvedCorpora:
    auto_mode = grouping_mode == "auto_single_cleaned"
    per_file = grouping_mode == "per_file"

    if auto_mode:
        group_files = resolve_auto_single_cleaned_group(
            cleaned_dir=cleaned_dir,
            group_name=config.grouping.auto_group_name,
        )
    else:
        group_files = resolve_group_files(
            groups=config.groups,
            project_root=project_root,
            cleaned_dir=cleaned_dir,
        )

    empty_groups = [name for name, files in group_files.items() if not files]
    if error_on_empty_group and empty_groups:
        names = ", ".join(empty_groups)
        raise CorpusPreparationError(f"No files matched for group(s): {names}")

    return ResolvedCorpora(
        cleaned_dir=cleaned_dir,
        group_files=group_files,
        work_items=build_corpus_work_items(group_files=group_files, group_by_file=per_file),
        mode=grouping_mode,
    )


def resolve_ref_tag_patterns(
    config: AppConfig,
    config_files: ResolvedConfigFiles,
) -> tuple[RefTagPattern, ...]:
    if not config.ref_tags.enabled:
        return ()
    path = config_files.path("ref_tags.patterns")
    if path is None:
        raise CorpusPreparationError(
            "ref_tags.patterns is required when ref_tags.enabled=true"
        )
    return tuple(load_ref_tag_patterns(path))


def prepare_corpus_text(
    *,
    work_item: CorpusWorkItem,
    config: AppConfig,
    ref_tag_patterns: Sequence[RefTagPattern] = (),
) -> PreparedCorpus:
    raw_text = read_concat(work_item.files)
    prepared_text = normalize_text(raw_text, config.normalization)
    ref_tag_counts: Counter[str] = Counter()
    if config.ref_tags.enabled:
        prepared_text, ref_tag_counts = strip_and_count_ref_tags(prepared_text, ref_tag_patterns)
    return PreparedCorpus(
        label=work_item.label,
        files=work_item.files,
        raw_text=raw_text,
        prepared_text=prepared_text,
        ref_tag_counts=ref_tag_counts,
    )


def prepare_corpora(
    *,
    work_items: Iterable[CorpusWorkItem],
    config: AppConfig,
    config_files: ResolvedConfigFiles,
) -> tuple[PreparedCorpus, ...]:
    ref_tag_patterns = resolve_ref_tag_patterns(config, config_files)
    return tuple(
        prepare_corpus_text(
            work_item=work_item,
            config=config,
            ref_tag_patterns=ref_tag_patterns,
        )
        for work_item in work_items
    )
