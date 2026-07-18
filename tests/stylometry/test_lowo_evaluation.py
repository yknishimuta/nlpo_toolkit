from __future__ import annotations

from pathlib import Path

import pytest

from nlpo_toolkit.stylometry.authorship import (
    build_work_profiles,
    label_feature_dataset,
    validate_lowo_dataset,
)
from nlpo_toolkit.stylometry.errors import StylometryError
from nlpo_toolkit.stylometry.evaluation import (
    build_leave_one_work_out_folds,
    evaluate_lowo,
)
from nlpo_toolkit.stylometry.evaluation_models import (
    AuthorshipAssignment,
    AuthorshipMetadata,
    WorkProfile,
)
from nlpo_toolkit.stylometry.metadata_reader import read_authorship_metadata
from nlpo_toolkit.stylometry.models import FeatureDataset, FeatureObservation


def test_metadata_reader_csv_tsv_custom_columns_and_order(tmp_path: Path) -> None:
    for input_format, delimiter in (("csv", ","), ("tsv", "\t")):
        path = tmp_path / f"metadata.{input_format}"
        path.write_text(
            delimiter.join(("writer", "id", "title"))
            + "\n"
            + delimiter.join((" A ", " s2 ", " w2 "))
            + "\n\n"
            + delimiter.join(("B", "s1", "w1"))
            + "\n",
            encoding="utf-8",
        )
        metadata = read_authorship_metadata(
            path,
            input_format=input_format,
            id_column="id",
            author_column="writer",
            work_column="title",
        )
        assert metadata.assignments == (
            AuthorshipAssignment("s2", "A", "w2"),
            AuthorshipAssignment("s1", "B", "w1"),
        )


@pytest.mark.parametrize(
    ("content", "message"),
    (
        ("id,author,work\n, a,w\n", "metadata ID is empty"),
        ("id,author,work\ns,,w\n", "author is empty"),
        ("id,author,work\ns,a,\n", "work ID is empty"),
        ("id,author,work\ns,a,w\ns,b,x\n", "duplicate metadata ID"),
        ("id,id,work\ns,a,w\n", "duplicate metadata column"),
        ("id,author\ns,a\n", "work column not found"),
        ("id,author,work\ns,a,w,extra\n", "wrong width"),
        ("id,author,work\ns,a,w\nt,b,w\n", "multiple authors"),
    ),
)
def test_metadata_reader_rejects_invalid_content(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(StylometryError, match=message):
        read_authorship_metadata(
            path,
            input_format="csv",
            id_column="id",
            author_column="author",
            work_column="work",
        )


def test_metadata_reader_reports_missing_empty_invalid_utf8(tmp_path: Path) -> None:
    kwargs = dict(
        input_format="csv", id_column="id", author_column="author", work_column="work"
    )
    with pytest.raises(StylometryError, match="metadata file not found"):
        read_authorship_metadata(tmp_path / "missing.csv", **kwargs)
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(StylometryError, match="no header"):
        read_authorship_metadata(empty, **kwargs)
    invalid = tmp_path / "invalid.csv"
    invalid.write_bytes(b"id,author,work\ns,a,\xff")
    with pytest.raises(StylometryError, match="not valid UTF-8"):
        read_authorship_metadata(invalid, **kwargs)


def _labeled_source() -> tuple[FeatureDataset, AuthorshipMetadata]:
    dataset = FeatureDataset(
        ("f1", "f2"),
        (
            FeatureObservation("a1", (0.0, 10.0)),
            FeatureObservation("a2", (2.0, 20.0)),
            FeatureObservation("b1", (8.0, 30.0)),
        ),
    )
    metadata = AuthorshipMetadata(
        (
            AuthorshipAssignment("extra", "X", "x"),
            AuthorshipAssignment("a1", "A", "work_a"),
            AuthorshipAssignment("a2", "A", "work_a"),
            AuthorshipAssignment("b1", "B", "work_b"),
        )
    )
    return dataset, metadata


def test_join_preserves_feature_order_and_work_profile_averages_samples() -> None:
    dataset, metadata = _labeled_source()
    labeled = label_feature_dataset(dataset, metadata=metadata)
    profiles = build_work_profiles(labeled)
    assert [item.identifier for item in labeled.observations] == ["a1", "a2", "b1"]
    assert profiles[0].work_id == "work_a"
    assert profiles[0].observation_ids == ("a1", "a2")
    assert profiles[0].values == (1.0, 15.0)
    assert profiles[1].values == (8.0, 30.0)
    with pytest.raises(StylometryError, match="has no authorship metadata"):
        label_feature_dataset(
            FeatureDataset(("f",), (FeatureObservation("missing", (1.0,)),)),
            metadata=metadata,
        )


def test_dataset_prerequisites_require_two_authors_and_two_works_each() -> None:
    one_author = FeatureDataset(
        ("f",),
        (
            FeatureObservation("a1", (0.0,)),
            FeatureObservation("a2", (1.0,)),
        ),
    )
    metadata = AuthorshipMetadata(
        (
            AuthorshipAssignment("a1", "A", "w1"),
            AuthorshipAssignment("a2", "A", "w2"),
        )
    )
    with pytest.raises(StylometryError, match="at least two authors"):
        validate_lowo_dataset(label_feature_dataset(one_author, metadata=metadata))

    dataset = FeatureDataset(
        ("f",),
        tuple(
            FeatureObservation(name, (float(index),))
            for index, name in enumerate(("a1", "a2", "b1"))
        ),
    )
    metadata = AuthorshipMetadata(
        (
            AuthorshipAssignment("a1", "A", "a1"),
            AuthorshipAssignment("a2", "A", "a2"),
            AuthorshipAssignment("b1", "B", "b1"),
        )
    )
    with pytest.raises(StylometryError, match="at least four works|only one work"):
        validate_lowo_dataset(label_feature_dataset(dataset, metadata=metadata))


def test_folds_hold_out_entire_work_and_preserve_order() -> None:
    dataset, metadata = _labeled_source()
    profiles = build_work_profiles(label_feature_dataset(dataset, metadata=metadata))
    folds = build_leave_one_work_out_folds(profiles)
    assert [fold.fold_index for fold in folds] == [1, 2]
    assert [fold.test_work.work_id for fold in folds] == ["work_a", "work_b"]
    assert folds[0].test_work.observation_ids == ("a1", "a2")
    assert [item.work_id for item in folds[0].training_works] == ["work_b"]


def test_hand_calculable_four_work_example_is_fully_correct() -> None:
    profiles = tuple(
        WorkProfile(work_id, author, (work_id,), (value,))
        for work_id, author, value in (
            ("A1", "A", 0.0),
            ("A2", "A", 1.0),
            ("B1", "B", 9.0),
            ("B2", "B", 11.0),
        )
    )
    result = evaluate_lowo(("f",), profiles)
    assert result.summary.work_count == 4
    assert result.summary.correct_work_count == 4
    assert result.summary.accuracy == 1.0
    assert result.summary.macro_author_accuracy == 1.0
    assert all(fold.is_correct for fold in result.folds)
    assert all(fold.margin >= 0.0 for fold in result.folds)


def test_each_fold_fits_only_training_work_profiles(monkeypatch) -> None:
    import nlpo_toolkit.stylometry.evaluation as evaluation_module

    profiles = tuple(
        WorkProfile(work, author, (f"{work}_sample",), (value,))
        for work, author, value in (
            ("A1", "A", 0.0),
            ("A2", "A", 1.0),
            ("B1", "B", 9.0),
            ("B2", "B", 1000.0),
        )
    )
    real_fit = evaluation_module.fit_zscore_model
    fitted_ids: list[tuple[str, ...]] = []

    def recording_fit(dataset):
        fitted_ids.append(tuple(item.identifier for item in dataset.observations))
        return real_fit(dataset)

    monkeypatch.setattr(evaluation_module, "fit_zscore_model", recording_fit)
    evaluate_lowo(("f",), profiles)

    assert fitted_ids == [
        ("A2", "B1", "B2"),
        ("A1", "B1", "B2"),
        ("A1", "A2", "B2"),
        ("A1", "A2", "B1"),
    ]


def test_all_zero_variance_training_fold_names_held_out_work() -> None:
    profiles = tuple(
        WorkProfile(work, author, (work,), (value,))
        for work, author, value in (
            ("A1", "A", 0.0),
            ("A2", "A", 0.0),
            ("B1", "B", 0.0),
            ("B2", "B", 1.0),
        )
    )
    with pytest.raises(StylometryError, match="held-out work 'B2'"):
        evaluate_lowo(("f",), profiles)
