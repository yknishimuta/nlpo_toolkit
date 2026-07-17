from __future__ import annotations

from collections.abc import Mapping

from ...postprocessing.dictionary import DictionaryClassification
from ..models import ArtifactKind, ArtifactPlan
from .frequency import (
    write_frequency_artifact,
    write_known_frequency_artifact,
    write_unknown_frequency_artifact,
)
from .reference_tags import write_reference_tags_artifact


def write_group_artifacts(
    *,
    artifact_plan: ArtifactPlan,
    group: str,
    counter: Mapping[str, int],
    dictionary: DictionaryClassification | None,
    reference_tag_counts: Mapping[str, int],
    csv_header: tuple[str, str],
    reference_tags_enabled: bool,
) -> None:
    write_frequency_artifact(
        artifact_plan.require(ArtifactKind.FREQUENCY, group=group),
        counter=counter,
        header=csv_header,
    )
    if dictionary is not None:
        write_known_frequency_artifact(
            artifact_plan.require(ArtifactKind.DICTCHECK_KNOWN, group=group),
            counter=dictionary.known,
            header=csv_header,
        )
        write_unknown_frequency_artifact(
            artifact_plan.require(ArtifactKind.DICTCHECK_UNKNOWN, group=group),
            counter=dictionary.unknown,
            header=csv_header,
        )
    if reference_tags_enabled:
        write_reference_tags_artifact(
            artifact_plan.require(ArtifactKind.REFERENCE_TAGS, group=group),
            counter=reference_tag_counts,
        )
