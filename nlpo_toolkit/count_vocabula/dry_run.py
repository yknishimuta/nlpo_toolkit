from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import load_config
from .io_utils import expand_globs
from .partition_validation import parse_partition_specs
from .preprocess import expand_cleaned_dir_placeholders, resolve_cleaner_output_dir
from .runner import _cleaned_txt_files


KNOWN_TOP_LEVEL_KEYS = {
    "analysis_unit",
    "archive",
    "cpu_only",
    "csv_header",
    "dictcheck",
    "filter",
    "filters",
    "group",
    "grouping",
    "groups",
    "language",
    "lemma_cache",
    "nlp",
    "normalization",
    "out_dir",
    "preprocess",
    "prune",
    "ref_tags",
    "stanza_package",
    "trace",
    "upos_targets",
    "validations",
    "vocab_path",
}

KNOWN_FILTER_KEYS = {
    "drop_roman_numerals",
    "min_token_length",
    "roman_exceptions_file",
}


class DuplicateKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: DuplicateKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> dict:
    mapping: dict[Any, Any] = {}
    duplicates = getattr(loader, "_duplicate_keys", [])

    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            duplicates.append(str(key))
        mapping[key] = loader.construct_object(value_node, deep=deep)

    loader._duplicate_keys = duplicates
    return mapping


DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _load_yaml_with_duplicate_keys(path: Path) -> tuple[dict[str, Any], list[str]]:
    loader = DuplicateKeyLoader(path.read_text(encoding="utf-8"))
    try:
        data = loader.get_single_data() or {}
        duplicates = list(getattr(loader, "_duplicate_keys", []))
    finally:
        loader.dispose()

    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping.")
    return data, duplicates


def _display_path(path: Path, project_root: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path)


def _resolve_project_path(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _warn_unknown_keys(raw_cfg: dict[str, Any], lines: list[str]) -> None:
    for key in raw_cfg:
        if key not in KNOWN_TOP_LEVEL_KEYS:
            lines.append(f"[WARN] unknown config key: {key}")

    filters = raw_cfg.get("filter") or raw_cfg.get("filters") or {}
    if isinstance(filters, dict):
        for key in filters:
            if key not in KNOWN_FILTER_KEYS:
                lines.append(f"[WARN] unknown config key: {key}")


def _cleaner_config_path(cfg: dict[str, Any], project_root: Path) -> Path | None:
    preprocess = cfg.get("preprocess")
    if not isinstance(preprocess, dict) or preprocess.get("kind") != "cleaner":
        return None

    raw_path = preprocess.get("config")
    if not raw_path:
        return None

    return _resolve_project_path(project_root, raw_path)


def _count_cleaner_input_files(cleaner_config_path: Path) -> int:
    cleaner_cfg = yaml.safe_load(cleaner_config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cleaner_cfg, dict):
        return 0

    raw_input = cleaner_cfg.get("input")
    if not raw_input:
        return 0

    input_path = Path(str(raw_input))
    if not input_path.is_absolute():
        input_path = (cleaner_config_path.parent / input_path).resolve()

    if input_path.is_file():
        return 1
    if input_path.is_dir():
        return len([p for p in input_path.rglob("*") if p.is_file()])

    return len(expand_globs([str(input_path)]))


def _group_files(cfg: dict[str, Any], project_root: Path, cleaned_dir: Path | None) -> dict[str, list[Path]]:
    groups = cfg.get("groups") or {}
    if not isinstance(groups, dict):
        return {}

    results: dict[str, list[Path]] = {}
    for group_name, group_def in groups.items():
        if not isinstance(group_def, dict):
            results[str(group_name)] = []
            continue

        patterns = group_def.get("files") or []
        if not isinstance(patterns, list):
            results[str(group_name)] = []
            continue

        resolved_patterns = [
            str(_resolve_project_path(project_root, p))
            if not Path(str(p)).is_absolute() and "{cleaned_dir}" not in str(p)
            else str(p)
            for p in patterns
        ]
        resolved_patterns = expand_cleaned_dir_placeholders(resolved_patterns, cleaned_dir)
        results[str(group_name)] = expand_globs(resolved_patterns)

    return results


def dry_run_count_vocabula(
    *,
    project_root: Path,
    config_path: Path,
    group_by_file: bool = False,
    error_on_empty_group: bool = False,
    auto_single_cleaned: bool = False,
) -> int:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    lines: list[str] = []
    exit_code = 0

    try:
        raw_cfg, duplicate_keys = _load_yaml_with_duplicate_keys(config_path)
        cfg = load_config(config_path)
        lines.append("[OK] config loaded")
    except Exception as exc:
        print(f"[ERROR] config: {exc}")
        return 1

    cleaner_config_path = _cleaner_config_path(cfg, project_root)
    cleaned_dir: Path | None = None
    if cleaner_config_path is not None:
        if cleaner_config_path.exists():
            lines.append(
                "[OK] preprocess cleaner config found: "
                f"{_display_path(cleaner_config_path, project_root)}"
            )
            try:
                cleaned_dir = resolve_cleaner_output_dir(cleaner_config_path)
                lines.append(f"[OK] input files: {_count_cleaner_input_files(cleaner_config_path)}")
                lines.append(f"[OK] cleaned output dir: {_display_path(cleaned_dir, project_root)}")
            except Exception as exc:
                lines.append(f"[ERROR] cleaner config: {exc}")
                exit_code = 1
        else:
            lines.append(
                "[ERROR] preprocess cleaner config missing: "
                f"{_display_path(cleaner_config_path, project_root)}"
            )
            exit_code = 1
    else:
        group_files = _group_files(cfg, project_root, None)
        lines.append(f"[OK] input files: {sum(len(files) for files in group_files.values())}")

    grouping = cfg.get("grouping") or {}
    grouping_mode = str(grouping.get("mode", "groups")).strip().lower()
    auto_mode = bool(auto_single_cleaned) or grouping_mode == "auto_single_cleaned"

    if auto_mode:
        auto_group_name = str(grouping.get("auto_group_name") or "text")
        try:
            cleaned_files = _cleaned_txt_files(cleaned_dir)
            if not cleaned_files:
                lines.append(
                    "[ERROR] --auto-single-cleaned was enabled, "
                    f"but no .txt files were found in {_display_path(cleaned_dir or Path('cleaned'), project_root)}"
                )
                group_files = {auto_group_name: []}
                exit_code = 1
            elif len(cleaned_files) > 1:
                lines.append(
                    "[ERROR] --auto-single-cleaned expected exactly one cleaned .txt file, "
                    f"but found {len(cleaned_files)}:"
                )
                for file_path in cleaned_files:
                    lines.append(f"  {_display_path(file_path, project_root)}")
                lines.append("")
                lines.append("Remove stale cleaned files, or specify groups.files explicitly.")
                group_files = {auto_group_name: cleaned_files}
                exit_code = 1
            else:
                lines.append("[OK] grouping mode: auto_single_cleaned")
                lines.append(
                    "[OK] auto selected cleaned file: "
                    f"{_display_path(cleaned_files[0], project_root)}"
                )
                group_files = {auto_group_name: cleaned_files}
        except ValueError as exc:
            lines.append(f"[ERROR] {exc}")
            group_files = {auto_group_name: []}
            exit_code = 1
    else:
        group_files = _group_files(cfg, project_root, cleaned_dir)

    for group_name, files in group_files.items():
        if not files and error_on_empty_group:
            lines.append(f"[ERROR] group {group_name} matched files: 0")
            exit_code = 1
        else:
            lines.append(f"[OK] group {group_name} matched files: {len(files)}")
        for file_path in files:
            lines.append(f"  - {_display_path(file_path, project_root)}")

    partition_specs = parse_partition_specs(cfg)
    if partition_specs and group_by_file:
        lines.append("[ERROR] validations.partitions cannot be used with --group-by-file")
        exit_code = 1
    for spec in partition_specs:
        empty_refs = [name for name in (spec.whole, *spec.parts) if not group_files.get(name)]
        if empty_refs:
            for group_name in empty_refs:
                lines.append(f"[ERROR] partition {spec.name} references empty group: {group_name}")
            exit_code = 1
        else:
            lines.append(
                f"[OK] partition {spec.name}: whole={spec.whole} parts={','.join(spec.parts)}"
            )

    for key in duplicate_keys:
        lines.append(f"[WARN] duplicate YAML key: {key}")
    _warn_unknown_keys(raw_cfg, lines)

    dictcheck = cfg.get("dictcheck") or {}
    if isinstance(dictcheck, dict) and dictcheck.get("enabled"):
        wordlist = dictcheck.get("wordlist")
        if wordlist:
            wordlist_path = _resolve_project_path(project_root, wordlist)
            if wordlist_path.exists():
                lines.append("[OK] dictcheck wordlist found")
            else:
                lines.append(f"[ERROR] dictcheck wordlist missing: {_display_path(wordlist_path, project_root)}")
                exit_code = 1

    ref_tags = cfg.get("ref_tags") or {}
    if isinstance(ref_tags, dict) and ref_tags.get("enabled"):
        patterns = ref_tags.get("patterns") or ref_tags.get("ref_tags_file")
        if patterns:
            patterns_path = _resolve_project_path(project_root, patterns)
            if patterns_path.exists():
                lines.append("[OK] ref_tags patterns found")
            else:
                lines.append(f"[ERROR] ref_tags patterns missing: {_display_path(patterns_path, project_root)}")
                exit_code = 1

    out_dir = _resolve_project_path(project_root, cfg.get("out_dir", "output"))
    lines.append(f"[OK] output dir: {_display_path(out_dir, project_root)}")

    mode = "auto_single_cleaned" if auto_mode else ("per_file" if group_by_file else grouping_mode)
    if mode == "per_file":
        lines.append("[OK] grouping mode: per_file")

    for line in lines:
        print(line)

    return exit_code
