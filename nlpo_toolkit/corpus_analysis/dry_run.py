from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import yaml

from .dependencies import CorpusPlanningDependencies
from .config import ConfigError
from .corpus import resolve_project_path
from .corpus_errors import CorpusPreparationError
from .run_plan import RunPlan, RunPlanError, build_run_plan

if TYPE_CHECKING:
    from .count_command import CountRequest


class DuplicateKeyLoader(yaml.SafeLoader):
    pass


class DryRunConfigError(ValueError):
    """The root configuration could not be inspected."""


@dataclass(frozen=True)
class DuplicateKeyConfig:
    raw: Mapping[str, object]
    duplicate_keys: tuple[str, ...]


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


def _load_yaml_with_duplicate_keys(path: Path) -> DuplicateKeyConfig:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DryRunConfigError(
            f"Failed to read config file: {path}: {exc}"
        ) from exc
    except UnicodeError as exc:
        raise DryRunConfigError(
            f"Config file is not valid UTF-8: {path}: {exc}"
        ) from exc

    loader = DuplicateKeyLoader(text)
    try:
        try:
            data = loader.get_single_data() or {}
            duplicates = tuple(getattr(loader, "_duplicate_keys", ()))
        except yaml.YAMLError as exc:
            raise DryRunConfigError(
                f"Invalid YAML in config file {path}: {exc}"
            ) from exc
    finally:
        loader.dispose()

    if not isinstance(data, dict):
        raise DryRunConfigError("Top-level YAML must be a mapping.")
    return DuplicateKeyConfig(raw=data, duplicate_keys=duplicates)


def _display_path(path: Path, project_root: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path)


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


def execute_dry_run(
    *,
    request: CountRequest,
    dependencies: CorpusPlanningDependencies,
) -> int:
    project_root = Path(request.project_root).resolve()
    config_path = Path(request.config_path)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    lines: list[str] = []
    exit_code = 0

    try:
        duplicate_config = _load_yaml_with_duplicate_keys(config_path)
        cfg = dependencies.load_config(config_path)
    except (DryRunConfigError, ConfigError) as exc:
        print(f"[ERROR] config: {exc}")
        return 1
    duplicate_keys = duplicate_config.duplicate_keys
    lines.append("[OK] config loaded")

    try:
        plan = build_run_plan(
            project_root=project_root,
            script_dir=None,
            config_path=config_path,
            group_by_file=request.group_by_file,
            auto_single_cleaned=request.auto_single_cleaned,
            error_on_empty_group=False,
            dependencies=CorpusPlanningDependencies(
                load_config=lambda _path: cfg,
                cleaner_loader=dependencies.cleaner_loader,
                cleaner_inspector=dependencies.cleaner_inspector,
            ),
            preprocess_mode="inspect",
            validate_references=False,
        )
    except (RunPlanError, CorpusPreparationError) as exc:
        lines.append(f"[ERROR] {exc}")
        exit_code = 1
    else:
        cleaner_inspection = plan.cleaner_inspection
        if cleaner_inspection is None:
            lines.append(
                f"[OK] input files: {sum(len(files) for files in plan.group_files.values())}"
            )
        else:
            lines.append(
                "[OK] preprocess cleaner config found: "
                f"{_display_path(cleaner_inspection.config.source_path, project_root)}"
            )
            lines.append(f"[OK] input files: {len(cleaner_inspection.input_files)}")
            lines.append(
                "[OK] cleaned output dir: "
                f"{_display_path(cleaner_inspection.config.output_path, project_root)}"
            )
        lines.extend(render_run_plan(plan, project_root=project_root))
        if request.error_on_empty_group:
            for group_name, files in plan.group_files.items():
                if not files:
                    lines.append(f"[ERROR] group {group_name} matched files: 0")
                    exit_code = 1
        spec_lines, spec_exit_code = _render_spec_diagnostics(plan)
        lines.extend(spec_lines)
        lines.append(f"[OK] output dir: {_display_path(plan.out_dir, project_root)}")
        exit_code = max(exit_code, spec_exit_code)

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
