from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import (
    load_config,
)
from .corpus import (
    inspect_preprocess,
    resolve_cleaner_plan,
    resolve_project_path,
)
from .io_utils import expand_globs
from .run_plan import RunPlan, build_run_plan


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


def render_run_plan(plan: RunPlan, *, project_root: Path) -> list[str]:
    lines: list[str] = []

    if plan.auto_mode:
        lines.append("[OK] grouping mode: auto_single_cleaned")
        files = plan.group_files.get(plan.auto_group_name, ())
        if files:
            lines.append(
                "[OK] auto selected cleaned file: "
                f"{_display_path(files[0], project_root)}"
            )
    elif plan.per_file:
        lines.append("[OK] grouping mode: per_file")

    for group_name, files in plan.group_files.items():
        lines.append(f"[OK] group {group_name} matched files: {len(files)}")
        for file_path in files:
            lines.append(f"  - {_display_path(file_path, project_root)}")

    return lines


def _render_spec_diagnostics(plan: RunPlan) -> tuple[list[str], int]:
    lines: list[str] = []
    exit_code = 0

    for spec in plan.partition_specs:
        empty_refs = [name for name in (spec.whole, *spec.parts) if not plan.group_files.get(name)]
        if empty_refs:
            for group_name in empty_refs:
                lines.append(f"[ERROR] partition {spec.name} references empty group: {group_name}")
            exit_code = 1
        else:
            lines.append(
                f"[OK] partition {spec.name}: whole={spec.whole} parts={','.join(spec.parts)}"
            )

    for spec in plan.comparison_specs:
        empty_refs = [name for name in (spec.group_a, spec.group_b) if not plan.group_files.get(name)]
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

    return lines, exit_code


def dry_run_count(
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
        _raw_cfg, duplicate_keys = _load_yaml_with_duplicate_keys(config_path)
        cfg = load_config(config_path)
        lines.append("[OK] config loaded")
    except Exception as exc:
        print(f"[ERROR] config: {exc}")
        return 1

    cleaner_plan = resolve_cleaner_plan(cfg, project_root)
    cleaner_config_path = cleaner_plan.config_path if cleaner_plan is not None else None
    if cleaner_config_path is not None:
        if cleaner_config_path.exists():
            lines.append(
                "[OK] preprocess cleaner config found: "
                f"{_display_path(cleaner_config_path, project_root)}"
            )
            try:
                cleaned_dir = inspect_preprocess(cleaner_plan)
                lines.append(f"[OK] input files: {_count_cleaner_input_files(cleaner_config_path)}")
                if cleaned_dir is not None:
                    lines.append(f"[OK] cleaned output dir: {_display_path(cleaned_dir, project_root)}")
            except Exception as exc:
                lines.append(f"[ERROR] {exc}")
                exit_code = 1
        else:
            lines.append(
                "[ERROR] preprocess cleaner config missing: "
                f"{_display_path(cleaner_config_path, project_root)}"
            )
            exit_code = 1

    try:
        plan = build_run_plan(
            project_root=project_root,
            script_dir=None,
            config_path=config_path,
            group_by_file=group_by_file,
            auto_single_cleaned=auto_single_cleaned,
            error_on_empty_group=False,
            load_config_fn=lambda _path: cfg,
            preprocess_mode="inspect",
            validate_references=False,
        )
        if cleaner_config_path is None:
            lines.append(f"[OK] input files: {sum(len(files) for files in plan.group_files.values())}")
        lines.extend(render_run_plan(plan, project_root=project_root))
        if error_on_empty_group:
            for group_name, files in plan.group_files.items():
                if not files:
                    lines.append(f"[ERROR] group {group_name} matched files: 0")
                    exit_code = 1
        spec_lines, spec_exit_code = _render_spec_diagnostics(plan)
        lines.extend(spec_lines)
        lines.append(f"[OK] output dir: {_display_path(plan.out_dir, project_root)}")
        exit_code = max(exit_code, spec_exit_code)
    except Exception as exc:
        lines.append(f"[ERROR] {exc}")
        exit_code = 1

    for key in duplicate_keys:
        lines.append(f"[WARN] duplicate YAML key: {key}")

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

    for line in lines:
        print(line)

    return exit_code
