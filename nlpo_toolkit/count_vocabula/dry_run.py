from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import (
    AppConfig,
    load_config,
    unknown_filter_keys,
    unknown_top_level_keys,
)
from .comparison import parse_comparison_specs
from .corpus import (
    cleaned_txt_files,
    inspect_preprocess,
    resolve_cleaner_plan,
    resolve_corpus_work_items,
    resolve_project_path,
)
from .io_utils import expand_globs
from .partition_validation import parse_partition_specs


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


def _warn_unknown_keys(raw_cfg: dict[str, Any], lines: list[str]) -> None:
    for key in unknown_top_level_keys(raw_cfg):
        lines.append(f"[WARN] unknown config key: {key}")
    for key in unknown_filter_keys(raw_cfg):
        lines.append(f"[WARN] unknown config key: {key}")


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

    grouping_mode = cfg.grouping.mode
    auto_mode = bool(auto_single_cleaned) or grouping_mode == "auto_single_cleaned"

    cleaner_plan = resolve_cleaner_plan(cfg, project_root)
    cleaner_config_path = cleaner_plan.config_path if cleaner_plan is not None else None
    cleaned_dir: Path | None = None
    if cleaner_config_path is not None:
        if cleaner_config_path.exists():
            lines.append(
                "[OK] preprocess cleaner config found: "
                f"{_display_path(cleaner_config_path, project_root)}"
            )
            try:
                cleaned_dir = inspect_preprocess(cleaner_plan)
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
        if auto_mode:
            group_files = {}
        else:
            resolved = resolve_corpus_work_items(
                config=cfg,
                project_root=project_root,
                cleaned_dir=None,
                group_by_file=group_by_file,
                auto_single_cleaned=False,
                error_on_empty_group=False,
            )
            group_files = resolved.group_files
        lines.append(f"[OK] input files: {sum(len(files) for files in group_files.values())}")

    if auto_mode:
        auto_group_name = cfg.grouping.auto_group_name
        try:
            cleaned_files = cleaned_txt_files(cleaned_dir)
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
        resolved = resolve_corpus_work_items(
            config=cfg,
            project_root=project_root,
            cleaned_dir=cleaned_dir,
            group_by_file=group_by_file,
            auto_single_cleaned=auto_single_cleaned,
            error_on_empty_group=False,
        )
        group_files = resolved.group_files

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

    comparison_specs = parse_comparison_specs(cfg)
    if comparison_specs and group_by_file:
        lines.append("[ERROR] comparisons cannot be used with grouping.mode=per_file")
        exit_code = 1
    for spec in comparison_specs:
        empty_refs = [name for name in (spec.group_a, spec.group_b) if not group_files.get(name)]
        if empty_refs:
            for group_name in empty_refs:
                lines.append(f"[ERROR] comparison {spec.name} references empty group: {group_name}")
            exit_code = 1
        else:
            lines.append(
                f"[OK] comparison {spec.name}: group_a={spec.group_a} "
                f"group_b={spec.group_b} scale={spec.scale} "
                f"min_total_count={spec.min_total_count}"
            )

    for key in duplicate_keys:
        lines.append(f"[WARN] duplicate YAML key: {key}")
    _warn_unknown_keys(raw_cfg, lines)

    if cfg.dictcheck.enabled:
        wordlist = cfg.dictcheck.wordlist
        if wordlist:
            wordlist_path = resolve_project_path(project_root, wordlist)
            if wordlist_path.exists():
                lines.append("[OK] dictcheck wordlist found")
            else:
                lines.append(f"[ERROR] dictcheck wordlist missing: {_display_path(wordlist_path, project_root)}")
                exit_code = 1

    if cfg.ref_tags.enabled:
        patterns = cfg.ref_tags.patterns
        if patterns:
            patterns_path = resolve_project_path(project_root, patterns)
            if patterns_path.exists():
                lines.append("[OK] ref_tags patterns found")
            else:
                lines.append(f"[ERROR] ref_tags patterns missing: {_display_path(patterns_path, project_root)}")
                exit_code = 1

    out_dir = resolve_project_path(project_root, cfg.out_dir)
    lines.append(f"[OK] output dir: {_display_path(out_dir, project_root)}")

    mode = "auto_single_cleaned" if auto_mode else ("per_file" if group_by_file else grouping_mode)
    if mode == "per_file":
        lines.append("[OK] grouping mode: per_file")

    for line in lines:
        print(line)

    return exit_code
