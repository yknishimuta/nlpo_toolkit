from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .ports import CorpusPlanningDependencies
from .config import ConfigError
from .config_references import ConfigReferenceError
from .corpus_errors import CorpusPreparationError
from .planning.build import build_count_plan
from .planning.models import ResolvedAnalysisPlan
from .planning.resolve import inspect_analysis_plan
from .planning.validate import AnalysisPlanError
from .requests import CorpusPreparationRequest


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


def _display_path(path: Path, project_root: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path)


def render_analysis_plan(plan: ResolvedAnalysisPlan, *, project_root: Path) -> list[str]:
    lines: list[str] = []

    definition = plan.definition
    if definition.auto_mode:
        lines.append("grouping mode: auto_single_cleaned")
        files = plan.group_files.get(definition.config.grouping.auto_group_name, ())
        if files:
            lines.append(
                "auto selected cleaned file: "
                f"{_display_path(files[0], project_root)}"
            )
    elif definition.per_file:
        lines.append("grouping mode: per_file")

    for group_name, files in plan.group_files.items():
        lines.append(f"group {group_name} matched files: {len(files)}")
        for file_path in files:
            lines.append(f"  - {_display_path(file_path, project_root)}")

    return lines


def _render_spec_diagnostics(plan: ResolvedAnalysisPlan) -> list[DryRunDiagnostic]:
    diagnostics: list[DryRunDiagnostic] = []

    config = plan.definition.config
    for spec in config.validations.partitions:
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

    for spec in config.comparisons:
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
    request: CorpusPreparationRequest,
    *,
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
        cfg = dependencies.load_config(config_path)
    except ConfigError as exc:
        return DryRunResult(
            diagnostics=(
                DryRunDiagnostic(DiagnosticLevel.ERROR, f"config: {exc}"),
            )
        )
    add(DiagnosticLevel.OK, "config loaded")

    try:
        definition = build_count_plan(
            request,
            dependencies=CorpusPlanningDependencies(
                load_config=lambda _path: cfg,
                cleaner_inspector=dependencies.cleaner_inspector,
            ),
        )
        plan = inspect_analysis_plan(definition)
    except (ConfigReferenceError, AnalysisPlanError, CorpusPreparationError) as exc:
        add(DiagnosticLevel.ERROR, str(exc))
    else:
        cleaner_inspection = (
            definition.cleaner_plan.inspection
            if definition.cleaner_plan is not None
            else None
        )
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
        diagnostics.extend(_render_spec_diagnostics(plan))
        add(DiagnosticLevel.OK, f"output dir: {_display_path(definition.out_dir, project_root)}")
        for reference in definition.config_files.references:
            if reference.kind == "root_config":
                continue
            add(
                DiagnosticLevel.OK,
                f"{reference.kind} found: "
                f"{_display_path(reference.source_path, project_root)}",
            )

    return DryRunResult(diagnostics=tuple(diagnostics))
