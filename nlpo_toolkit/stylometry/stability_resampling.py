from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Sequence

from .evaluation_models import LabeledFeatureDataset, WorkProfile
from .errors import StylometryError


def derive_iteration_seed(root_seed: int, iteration: int, attempt: int) -> int:
    value = (
        "stylometry-verification-stability-v1\n"
        f"seed={root_seed}\niteration={iteration}\nattempt={attempt}\n"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big", signed=False)


def stable_feature_hash(names: Sequence[str]) -> str:
    canonical = json.dumps(
        tuple(names), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def subsample_reference_works(
    profiles: Sequence[WorkProfile],
    *,
    fraction: float,
    minimum: int,
    rng: random.Random,
) -> tuple[WorkProfile, ...]:
    ordered = tuple(sorted(profiles, key=lambda item: item.work_id))
    size = min(len(ordered), max(minimum, math.ceil(len(ordered) * fraction)))
    selected_ids = {item.work_id for item in rng.sample(ordered, size)}
    return tuple(item for item in ordered if item.work_id in selected_ids)


def bootstrap_work_profiles(
    dataset: LabeledFeatureDataset,
    *,
    included_work_ids: frozenset[str],
    rng: random.Random,
) -> tuple[WorkProfile, ...]:
    grouped = {}
    for observation in dataset.observations:
        if observation.work_id in included_work_ids:
            grouped.setdefault(observation.work_id, []).append(observation)
    profiles = []
    for work_id, observations in grouped.items():
        sampled = tuple(rng.choice(observations) for _ in observations)
        profiles.append(
            WorkProfile(
                work_id,
                observations[0].author,
                tuple(item.identifier for item in sampled),
                tuple(
                    sum(item.values[index] for item in sampled) / len(sampled)
                    for index in range(dataset.feature_count)
                ),
            )
        )
    return tuple(profiles)


def subsample_feature_names(
    feature_names: Sequence[str],
    *,
    fraction: float,
    rng: random.Random,
) -> tuple[str, ...]:
    names = tuple(feature_names)
    if not names:
        raise StylometryError("feature subsampling requires feature names")
    size = max(1, math.ceil(len(names) * fraction))
    selected_indices = set(rng.sample(range(len(names)), size))
    return tuple(name for index, name in enumerate(names) if index in selected_indices)


def select_profile_features(
    profiles: Sequence[WorkProfile],
    *,
    input_feature_names: tuple[str, ...],
    selected_feature_names: tuple[str, ...],
) -> tuple[WorkProfile, ...]:
    indices = tuple(input_feature_names.index(name) for name in selected_feature_names)
    return tuple(
        WorkProfile(
            item.work_id,
            item.author,
            item.observation_ids,
            tuple(item.values[index] for index in indices),
        )
        for item in profiles
    )
