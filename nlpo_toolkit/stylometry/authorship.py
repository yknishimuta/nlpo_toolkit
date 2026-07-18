from __future__ import annotations

from .errors import StylometryError
from .evaluation_models import (
    AuthorshipMetadata,
    LabeledFeatureDataset,
    LabeledFeatureObservation,
    WorkProfile,
)
from .models import FeatureDataset


def label_feature_dataset(
    dataset: FeatureDataset, *, metadata: AuthorshipMetadata
) -> LabeledFeatureDataset:
    assignments = {item.observation_id: item for item in metadata.assignments}
    labeled = []
    for observation in dataset.observations:
        assignment = assignments.get(observation.identifier)
        if assignment is None:
            raise StylometryError(
                f"feature observation {observation.identifier!r} has no authorship metadata"
            )
        labeled.append(
            LabeledFeatureObservation(
                observation.identifier,
                assignment.author,
                assignment.work_id,
                observation.values,
            )
        )
    return LabeledFeatureDataset(dataset.feature_names, tuple(labeled))


def validate_lowo_dataset(dataset: LabeledFeatureDataset) -> None:
    author_works: dict[str, list[str]] = {}
    for item in dataset.observations:
        works = author_works.setdefault(item.author, [])
        if item.work_id not in works:
            works.append(item.work_id)
    if len(author_works) < 2:
        raise StylometryError("leave-one-work-out requires at least two authors")
    if dataset.work_count < 4:
        raise StylometryError("leave-one-work-out requires at least four works")
    for author, works in author_works.items():
        if len(works) < 2:
            raise StylometryError(
                "leave-one-work-out requires at least two works per author; "
                f"author {author!r} has only one work"
            )


def build_work_profiles(dataset: LabeledFeatureDataset) -> tuple[WorkProfile, ...]:
    grouped: dict[str, list[LabeledFeatureObservation]] = {}
    for item in dataset.observations:
        grouped.setdefault(item.work_id, []).append(item)
    profiles = []
    for work_id, items in grouped.items():
        values = tuple(
            sum(item.values[index] for item in items) / len(items)
            for index in range(dataset.feature_count)
        )
        profiles.append(
            WorkProfile(
                work_id=work_id,
                author=items[0].author,
                observation_ids=tuple(item.identifier for item in items),
                values=values,
            )
        )
    return tuple(profiles)
