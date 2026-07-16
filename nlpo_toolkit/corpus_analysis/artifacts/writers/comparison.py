from __future__ import annotations

from typing import Mapping, Sequence

from nlpo_toolkit.comparison.configured import ComparisonResult
from nlpo_toolkit.comparison.writers import comparison_result_summary

from ..models import ArtifactKind, PlannedArtifact
from ..publication import ArtifactPublicationError, open_csv_artifact, publish_json


_FIELDS = ("comparison", "analysis_unit", "item", "group_a", "group_b", "group_a_count", "group_b_count", "group_a_tokens", "group_b_tokens", "scale", "group_a_rate", "group_b_rate", "rate_difference", "log_ratio", "log_likelihood", "direction", "total_count")


def _require(artifact: PlannedArtifact, kind: ArtifactKind) -> None:
    if artifact.kind is not kind:
        raise ArtifactPublicationError(f"Wrong artifact kind: expected={kind.value} actual={artifact.kind.value} name={artifact.name} path={artifact.path}")


def write_comparison_csv_artifact(artifact: PlannedArtifact, *, result: ComparisonResult) -> None:
    _require(artifact, ArtifactKind.GROUP_COMPARISON_CSV)
    with open_csv_artifact(artifact, fieldnames=_FIELDS) as writer:
        writer.writeheader()
        for row in result.rows:
            writer.writerow({"comparison": result.spec.name, "analysis_unit": result.analysis_unit, "item": row.item, "group_a": result.spec.group_a, "group_b": result.spec.group_b, "group_a_count": row.group_a_count, "group_b_count": row.group_b_count, "group_a_tokens": row.group_a_tokens, "group_b_tokens": row.group_b_tokens, "scale": row.scale, "group_a_rate": f"{row.group_a_rate:.6f}", "group_b_rate": f"{row.group_b_rate:.6f}", "rate_difference": f"{row.rate_difference:.6f}", "log_ratio": f"{row.log_ratio:.6f}", "log_likelihood": f"{row.log_likelihood:.6f}", "direction": row.direction, "total_count": row.total_count})


def render_comparisons_json(results: Sequence[ComparisonResult], *, csv_names: Mapping[str, str]) -> dict[str, object]:
    return {"analysis_unit": results[0].analysis_unit if results else "", "comparisons": [comparison_result_summary(result, csv_name=csv_names[result.spec.name]) for result in results]}


def write_comparisons_json_artifact(artifact: PlannedArtifact, *, data: object) -> None:
    _require(artifact, ArtifactKind.GROUP_COMPARISONS_JSON)
    publish_json(artifact, data=data)
