from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from nlpo_toolkit.stylometry.delta import build_delta_pairs, burrows_delta
from nlpo_toolkit.stylometry.errors import StylometryError
from nlpo_toolkit.stylometry.models import (
    FeatureDataset,
    FeatureObservation,
    FeatureSelection,
    StandardizedObservation,
)
from nlpo_toolkit.stylometry.standardization import (
    fit_zscore_model,
    transform_feature_dataset,
)


def _reference_dataset() -> FeatureDataset:
    return FeatureDataset(
        ("f1", "f2"),
        (
            FeatureObservation("A", (1.0, 10.0)),
            FeatureObservation("B", (2.0, 20.0)),
            FeatureObservation("C", (3.0, 10.0)),
        ),
    )


def test_selection_normalizes_tuples_and_validates_names() -> None:
    selection = FeatureSelection("sample", ["mfw_"], ["fw_et"])
    assert selection.prefixes == ("mfw_",)
    assert selection.columns == ("fw_et",)
    with pytest.raises(FrozenInstanceError):
        selection.id_column = "other"  # type: ignore[misc]
    for kwargs in (
        {"id_column": "", "prefixes": ("x",)},
        {"prefixes": ("",)},
        {"columns": ("",)},
        {"prefixes": ("x", "x")},
        {"columns": ("x", "x")},
        {},
    ):
        with pytest.raises(StylometryError):
            FeatureSelection(**kwargs)


def test_dataset_is_deeply_immutable_and_validated() -> None:
    observation = FeatureObservation("A", [1, 2])
    dataset = FeatureDataset(["f1", "f2"], [observation])
    assert observation.values == (1, 2)
    assert dataset.feature_names == ("f1", "f2")
    assert dataset.observations == (observation,)
    assert dataset.sample_count == 1
    assert dataset.feature_count == 2
    with pytest.raises(StylometryError):
        FeatureObservation("", (1.0,))
    with pytest.raises(StylometryError):
        FeatureObservation("A", (math.nan,))
    with pytest.raises(StylometryError):
        FeatureObservation("A", (math.inf,))
    with pytest.raises(StylometryError):
        FeatureObservation("A", (True,))
    with pytest.raises(StylometryError):
        FeatureDataset(("f", "f"), ())
    with pytest.raises(StylometryError):
        FeatureDataset(("f",), (FeatureObservation("A", (1.0, 2.0)),))
    with pytest.raises(StylometryError):
        FeatureDataset(
            ("f",),
            (FeatureObservation("A", (1.0,)), FeatureObservation("A", (2.0,))),
        )


def test_fit_uses_sample_standard_deviation_and_transform_matches_reference() -> None:
    dataset = _reference_dataset()
    model = fit_zscore_model(dataset)
    standardized = transform_feature_dataset(dataset, model=model)

    assert model.means == pytest.approx((2.0, 40 / 3))
    assert model.standard_deviations == pytest.approx((1.0, math.sqrt(100 / 3)))
    assert standardized.observations[0].values == pytest.approx(
        (-1.0, -1 / math.sqrt(3))
    )
    assert standardized.observations[1].values == pytest.approx((0.0, 2 / math.sqrt(3)))
    assert standardized.observations[2].values == pytest.approx(
        (1.0, -1 / math.sqrt(3))
    )


def test_fit_drops_exact_zero_variance_in_header_order() -> None:
    dataset = FeatureDataset(
        ("constant_a", "varying", "constant_b"),
        (
            FeatureObservation("A", (1.0, 1.0, 4.0)),
            FeatureObservation("B", (1.0, 2.0, 4.0)),
        ),
    )
    model = fit_zscore_model(dataset)
    assert model.retained_feature_names == ("varying",)
    assert model.retained_indices == (1,)
    assert model.dropped_zero_variance_features == ("constant_a", "constant_b")
    with pytest.raises(StylometryError, match="all selected features"):
        fit_zscore_model(
            FeatureDataset(
                ("constant",),
                (
                    FeatureObservation("A", (1.0,)),
                    FeatureObservation("B", (1.0,)),
                ),
            )
        )


def test_transform_rejects_schema_mismatch() -> None:
    dataset = _reference_dataset()
    model = fit_zscore_model(dataset)
    reordered = FeatureDataset(
        ("f2", "f1"),
        tuple(
            FeatureObservation(item.identifier, tuple(reversed(item.values)))
            for item in dataset.observations
        ),
    )
    with pytest.raises(StylometryError, match="schema"):
        transform_feature_dataset(reordered, model=model)


def test_delta_reference_pairs_and_sorting() -> None:
    model = fit_zscore_model(_reference_dataset())
    standardized = transform_feature_dataset(_reference_dataset(), model=model)
    pairs = build_delta_pairs(standardized)

    assert len(pairs) == 3
    assert [(pair.sample_a, pair.sample_b) for pair in pairs] == [
        ("A", "C"),
        ("A", "B"),
        ("B", "C"),
    ]
    assert pairs[0].distance == pytest.approx(1.0)
    assert pairs[1].distance == pytest.approx((1 + math.sqrt(3)) / 2)
    assert pairs[2].distance == pytest.approx((1 + math.sqrt(3)) / 2)


def test_delta_is_symmetric_non_negative_and_zero_for_self() -> None:
    first = StandardizedObservation("A", (-1.0, 2.0))
    second = StandardizedObservation("B", (1.0, 0.0))
    assert burrows_delta(first, first) == 0.0
    assert burrows_delta(first, second) == burrows_delta(second, first) == 2.0
