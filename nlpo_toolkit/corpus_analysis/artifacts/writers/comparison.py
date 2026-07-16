from __future__ import annotations

from typing import Mapping, Sequence

from nlpo_toolkit.comparison.config import ComparisonSpec
from nlpo_toolkit.comparison.engine import FrequencyTable
from nlpo_toolkit.comparison.results import ConfiguredComparisonResult

from ..models import ArtifactKind, PlannedArtifact
from ..publication import ArtifactPublicationError, open_csv_artifact, publish_json


ConfiguredResult = ConfiguredComparisonResult[ComparisonSpec, FrequencyTable]
_FIELDS = ("comparison", "analysis_unit", "item", "group_a", "group_b", "group_a_count", "group_b_count", "group_a_tokens", "group_b_tokens", "scale", "group_a_rate", "group_b_rate", "rate_difference", "log_ratio", "log_likelihood", "direction", "total_count")


def _require(artifact: PlannedArtifact, kind: ArtifactKind) -> None:
    if artifact.kind is not kind:
        raise ArtifactPublicationError(f"Wrong artifact kind: expected={kind.value} actual={artifact.kind.value} name={artifact.name} path={artifact.path}")


def write_comparison_csv_artifact(artifact: PlannedArtifact, *, result: ConfiguredResult) -> None:
    _require(artifact, ArtifactKind.GROUP_COMPARISON_CSV)
    with open_csv_artifact(artifact, fieldnames=_FIELDS) as writer:
        writer.writeheader()
        for row in result.rows:
            writer.writerow({"comparison": result.spec.name, "analysis_unit": result.analysis_unit, "item": row.item, "group_a": result.spec.group_a, "group_b": result.spec.group_b, "group_a_count": int(row.count_a), "group_b_count": int(row.count_b), "group_a_tokens": int(result.group_a_tokens), "group_b_tokens": int(result.group_b_tokens), "scale": result.spec.scale, "group_a_rate": f"{row.rate_a:.6f}", "group_b_rate": f"{row.rate_b:.6f}", "rate_difference": f"{row.rate_difference:.6f}", "log_ratio": f"{row.log_ratio:.6f}", "log_likelihood": f"{row.log_likelihood:.6f}", "direction": row.direction, "total_count": int(row.total_count)})


def _summary(result: ConfiguredResult, *, csv_name: str) -> dict[str, object]:
    spec = result.spec
    return {"name": spec.name, "group_a": spec.group_a, "group_b": spec.group_b, "scale": spec.scale, "zero_correction": spec.zero_correction, "min_total_count": spec.min_total_count, "group_a_tokens": int(result.group_a_tokens), "group_b_tokens": int(result.group_b_tokens), "vocabulary_union_size": result.vocabulary_union_size, "rows_before_filter": result.rows_before_filter, "rows_after_filter": result.rows_after_filter, "csv": csv_name}


def render_comparisons_json(results: Sequence[ConfiguredResult], *, csv_names: Mapping[str, str]) -> dict[str, object]:
    return {"analysis_unit": results[0].analysis_unit if results else "", "comparisons": [_summary(result, csv_name=csv_names[result.spec.name]) for result in results]}


def write_comparisons_json_artifact(artifact: PlannedArtifact, *, data: object) -> None:
    _require(artifact, ArtifactKind.GROUP_COMPARISONS_JSON)
    publish_json(artifact, data=data)
