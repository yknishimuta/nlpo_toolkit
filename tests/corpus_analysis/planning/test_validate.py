from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.config import ensure_app_config
from nlpo_toolkit.corpus_analysis.config_references import ResolvedConfigFiles
from nlpo_toolkit.corpus_analysis.planning.models import AnalysisMode, AnalysisPlan, ResolvedAnalysisPlan
from nlpo_toolkit.corpus_analysis.planning.validate import (
    AnalysisPlanError,
    validate_count_group_references,
    validate_count_plan_structure,
)


def _plan(tmp_path: Path, config_data: dict, *, mode="groups", groups=None):
    config = ensure_app_config(config_data)
    definition = AnalysisPlan(
        tmp_path, tmp_path / "config.yml", config, tmp_path / "out", mode, False,
        AnalysisMode(config.analysis_unit, ("lemma", "count")), None, ResolvedConfigFiles(),
    )
    return definition, ResolvedAnalysisPlan(definition, None, (), groups or {})


def test_partition_and_comparison_reject_per_file(tmp_path: Path) -> None:
    partition, _ = _plan(tmp_path, {
        "groups": {"whole": {"files": []}, "part": {"files": []}, "other": {"files": []}},
        "validations": {"partitions": [{"name": "p", "whole": "whole", "parts": ["part", "other"]}]},
    }, mode="per_file")
    with pytest.raises(AnalysisPlanError, match="partitions"):
        validate_count_plan_structure(partition)
    comparison, _ = _plan(tmp_path, {
        "groups": {"a": {"files": []}, "b": {"files": []}},
        "comparisons": [{"name": "c", "group_a": "a", "group_b": "b"}],
    }, mode="per_file")
    with pytest.raises(AnalysisPlanError, match="comparisons"):
        validate_count_plan_structure(comparison)


@pytest.mark.parametrize("missing", ["whole", "part"])
def test_partition_empty_reference_is_rejected(tmp_path: Path, missing: str) -> None:
    _, resolved = _plan(tmp_path, {
        "groups": {"whole": {"files": []}, "part": {"files": []}, "other": {"files": []}},
        "validations": {"partitions": [{"name": "p", "whole": "whole", "parts": ["part", "other"]}]},
    }, groups={"whole": () if missing == "whole" else (tmp_path / "w",), "part": () if missing == "part" else (tmp_path / "p",), "other": (tmp_path / "o",)})
    with pytest.raises(AnalysisPlanError, match=f"empty group: {missing}"):
        validate_count_group_references(resolved)


def test_non_empty_references_are_valid(tmp_path: Path) -> None:
    definition, resolved = _plan(tmp_path, {
        "groups": {"a": {"files": []}, "b": {"files": []}},
        "comparisons": [{"name": "c", "group_a": "a", "group_b": "b"}],
    }, groups={"a": (tmp_path / "a",), "b": (tmp_path / "b",)})
    validate_count_plan_structure(definition)
    validate_count_group_references(resolved)
