from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping

import yaml

from .dependencies import CorpusPlanningDependencies
from .config import ConfigError
from .config_references import ConfigReferenceError
from .corpus_errors import CorpusPreparationError
from .run_plan import AnalysisPlan, AnalysisPlanError, build_count_plan

if TYPE_CHECKING:
    from .count_command import CountRequest


class DuplicateKeyLoader(yaml.SafeLoader):
    pass


class DryRunConfigError(ValueError):
    """The root configuration could not be inspected."""


class DiagnosticLevel(Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class DryRunDiagnostic:
    level: DiagnosticLevel
    message: str


@dataclass(frozen=True)
class DryRunResult:
    diagnostics: tuple[DryRunDiagnostic, ...]

    @property
    def successful(self) -> bool:
        return not any(
            item.level is DiagnosticLevel.ERROR for item in self.diagnostics
        )


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


def render_analysis_plan(plan: AnalysisPlan, *, project_root: Path) -> list[str]:
    lines: list[str] = []

    if plan.auto_mode:
        lines.append("grouping mode: auto_single_cleaned")
        files = plan.group_files.get(plan.auto_group_name, ())
        if files:
            lines.append(
                "auto selected cleaned file: "
                f"{_display_path(files[0], project_root)}"
            )
    elif plan.per_file:
        lines.append("grouping mode: per_file")

    for group_name, files in plan.group_files.items():
        lines.append(f"group {group_name} matched files: {len(files)}")
        for file_path in files:
            lines.append(f"  - {_display_path(file_path, project_root)}")

    return lines


def _render_spec_diagnostics(plan: AnalysisPlan) -> list[DryRunDiagnostic]:
    diagnostics: list[DryRunDiagnostic] = []

    for spec in plan.partition_specs:
        empty_refs = [name for name in (spec.whole, *spec.parts) if not plan.group_files.get(name)]
        if empty_refs:
            for group_name in empty_refs:
                diagnostics.append(
                    DryRunDiagnostic(
                        DiagnosticLevel.ERROR,
                        f"partition {spec.name} references empty group: {group_name}",
                    )
                )
        else:
            diagnostics.append(
                DryRunDiagnostic(
                    DiagnosticLevel.OK,
                    f"partition {spec.name}: whole={spec.whole} parts={','.join(spec.parts)}",
                )
            )

    for spec in plan.comparison_specs:
        empty_refs = [name for name in (spec.group_a, spec.group_b) if not plan.group_files.get(name)]
        if empty_refs:
            for group_name in empty_refs:
                diagnostics.append(
                    DryRunDiagnostic(
                        DiagnosticLevel.ERROR,
                        f"comparison {spec.name} references empty group: {group_name}",
                    )
                )
        else:
            diagnostics.append(
                DryRunDiagnostic(
                    DiagnosticLevel.OK,
                    f"comparison {spec.name}: group_a={spec.group_a} "
                    f"group_b={spec.group_b} scale={spec.scale} "
                    f"min_total_count={spec.min_total_count}",
                )
            )

    return diagnostics


def execute_dry_run(
    *,
    request: CountRequest,
    dependencies: CorpusPlanningDependencies,
) -> DryRunResult:
    project_root = Path(request.project_root).resolve()
    config_path = Path(request.config_path)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    diagnostics: list[DryRunDiagnostic] = []

    def add(level: DiagnosticLevel, message: str) -> None:
        diagnostics.append(DryRunDiagnostic(level, message))

    try:
        duplicate_config = _load_yaml_with_duplicate_keys(config_path)
        cfg = dependencies.load_config(config_path)
    except (DryRunConfigError, ConfigError) as exc:
        return DryRunResult(
            diagnostics=(
                DryRunDiagnostic(DiagnosticLevel.ERROR, f"config: {exc}"),
            )
        )
    duplicate_keys = duplicate_config.duplicate_keys
    add(DiagnosticLevel.OK, "config loaded")

    try:
        plan = build_count_plan(
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
    except (ConfigReferenceError, AnalysisPlanError, CorpusPreparationError) as exc:
        add(DiagnosticLevel.ERROR, str(exc))
    else:
        cleaner_inspection = plan.cleaner_inspection
        if cleaner_inspection is None:
            add(
                DiagnosticLevel.OK,
                f"input files: {sum(len(files) for files in plan.group_files.values())}",
            )
        else:
            add(
                DiagnosticLevel.OK,
                "preprocess cleaner config found: "
                f"{_display_path(cleaner_inspection.config.source_path, project_root)}"
            )
            add(DiagnosticLevel.OK, f"input files: {len(cleaner_inspection.input_files)}")
            add(
                DiagnosticLevel.OK,
                "cleaned output dir: "
                f"{_display_path(cleaner_inspection.config.output_path, project_root)}"
            )
        for line in render_analysis_plan(plan, project_root=project_root):
            add(DiagnosticLevel.OK, line)
        if request.error_on_empty_group:
            for group_name, files in plan.group_files.items():
                if not files:
                    add(DiagnosticLevel.ERROR, f"group {group_name} matched files: 0")
        diagnostics.extend(_render_spec_diagnostics(plan))
        add(DiagnosticLevel.OK, f"output dir: {_display_path(plan.out_dir, project_root)}")
        for reference in plan.config_files.references:
            if reference.kind == "root_config":
                continue
            add(
                DiagnosticLevel.OK,
                f"{reference.kind} found: "
                f"{_display_path(reference.path, project_root)}",
            )

    for key in duplicate_keys:
        add(DiagnosticLevel.WARNING, f"duplicate YAML key: {key}")

    return DryRunResult(diagnostics=tuple(diagnostics))
