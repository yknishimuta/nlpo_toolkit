from __future__ import annotations

from typing import Sequence

from ...partition_models import PartitionSpec
from ...partition_validation import PartitionResult, partition_result_summary
from ..models import ArtifactKind, PlannedArtifact
from ..publication import ArtifactPublicationError, open_csv_artifact, publish_json


def _require(artifact: PlannedArtifact, kind: ArtifactKind) -> None:
    if artifact.kind is not kind:
        raise ArtifactPublicationError(
            f"Wrong artifact kind: expected={kind.value} actual={artifact.kind.value} "
            f"name={artifact.name} path={artifact.path}"
        )


def write_partition_csv_artifact(artifact: PlannedArtifact, *, result: PartitionResult) -> None:
    _require(artifact, ArtifactKind.PARTITION_VALIDATION_CSV)
    fields = ["item", "whole_count", *(f"{part}_count" for part in result.parts), "parts_sum", "delta", "status"]
    with open_csv_artifact(artifact, fieldnames=fields) as writer:
        writer.writeheader()
        for row in result.mismatches:
            data = {"item": row.item, "whole_count": row.whole_count, "parts_sum": row.parts_sum, "delta": row.delta, "status": row.status}
            data.update({f"{part}_count": row.part_counts.get(part, 0) for part in result.parts})
            writer.writerow(data)


def render_partition_json(specs: Sequence[PartitionSpec], results: Sequence[PartitionResult], *, csv_names: dict[str, str]) -> dict[str, object]:
    return {"partitions": [partition_result_summary(spec, result, csv_name=csv_names[spec.name]) for spec, result in zip(specs, results)]}


def write_partition_json_artifact(artifact: PlannedArtifact, *, data: object) -> None:
    _require(artifact, ArtifactKind.PARTITION_VALIDATION_JSON)
    publish_json(artifact, data=data)
