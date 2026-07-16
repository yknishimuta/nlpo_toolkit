from __future__ import annotations

from ..config.models import AppConfig
from .models import AnalysisPlan, ResolvedAnalysisPlan


class AnalysisPlanError(ValueError):
    pass


def validate_analysis_config(config: AppConfig) -> None:
    if not config.groups:
        raise AnalysisPlanError("config.groups must be a non-empty mapping")


def validate_count_plan_structure(plan: AnalysisPlan) -> None:
    if plan.config.validations.partitions and plan.per_file:
        raise AnalysisPlanError(
            "validations.partitions cannot be used with --group-by-file or "
            "grouping.mode: per_file"
        )
    if plan.config.comparisons and plan.per_file:
        raise AnalysisPlanError(
            "comparisons cannot be used with grouping.mode=per_file"
        )


def validate_count_group_references(plan: ResolvedAnalysisPlan) -> None:
    config = plan.definition.config
    for spec in config.validations.partitions:
        for name in (spec.whole, *spec.parts):
            if not plan.group_files.get(name):
                raise AnalysisPlanError(
                    f"partition {spec.name} references empty group: {name}"
                )
    for spec in config.comparisons:
        for name in (spec.group_a, spec.group_b):
            if not plan.group_files.get(name):
                raise AnalysisPlanError(
                    f"comparison {spec.name} references empty group: {name}"
                )
