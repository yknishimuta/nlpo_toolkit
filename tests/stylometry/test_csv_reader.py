from __future__ import annotations

from pathlib import Path

import pytest

from nlpo_toolkit.stylometry.csv_reader import read_feature_dataset
from nlpo_toolkit.stylometry.errors import StylometryError
from nlpo_toolkit.stylometry.models import FeatureSelection


def test_reader_selects_header_order_union_for_csv_and_tsv(tmp_path: Path) -> None:
    for input_format, delimiter in (("csv", ","), ("tsv", "\t")):
        path = tmp_path / f"features.{input_format}"
        path.write_text(
            delimiter.join(("group", "ignored", "mfw_b", "fw_et", "mfw_a"))
            + "\n"
            + delimiter.join(("A", "99", "2", "1", "3"))
            + "\n\n"
            + delimiter.join(("B", "88", "4", "2", "6"))
            + "\n",
            encoding="utf-8",
        )
        dataset = read_feature_dataset(
            path,
            input_format=input_format,
            selection=FeatureSelection(
                id_column="group", prefixes=("mfw_",), columns=("fw_et", "mfw_b")
            ),
        )
        assert dataset.feature_names == ("mfw_b", "fw_et", "mfw_a")
        assert dataset.observations[0].values == (2.0, 1.0, 3.0)


@pytest.mark.parametrize(
    ("content", "selection", "message"),
    (
        (
            "group,f,f\nA,1,2\nB,2,3\n",
            FeatureSelection(prefixes=("f",)),
            "duplicate feature-table",
        ),
        (
            "other,f\nA,1\nB,2\n",
            FeatureSelection(prefixes=("f",)),
            "identifier column not found",
        ),
        (
            "group,f\nA,1\nB,2\n",
            FeatureSelection(columns=("missing",)),
            "feature column not found",
        ),
        (
            "group,f\nA,1\nB,2\n",
            FeatureSelection(prefixes=("x",)),
            "no feature columns",
        ),
        ("group,f\nA,\nB,2\n", FeatureSelection(prefixes=("f",)), "value is empty"),
        ("group,f\nA,x\nB,2\n", FeatureSelection(prefixes=("f",)), "not numeric"),
        ("group,f\nA,NaN\nB,2\n", FeatureSelection(prefixes=("f",)), "must be finite"),
        (
            "group,f\nA,Infinity\nB,2\n",
            FeatureSelection(prefixes=("f",)),
            "must be finite",
        ),
        ("group,f\nA,1\nA,2\n", FeatureSelection(prefixes=("f",)), "duplicate value"),
        ("group,f\nA,1,extra\nB,2\n", FeatureSelection(prefixes=("f",)), "wrong width"),
        (
            "group,f\nA,1\n",
            FeatureSelection(prefixes=("f",)),
            "at least two observations",
        ),
    ),
)
def test_reader_rejects_invalid_tables(
    tmp_path: Path, content: str, selection: FeatureSelection, message: str
) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(StylometryError, match=message):
        read_feature_dataset(path, input_format="csv", selection=selection)


def test_reader_rejects_id_as_explicit_feature(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    path.write_text("group,f\nA,1\nB,2\n", encoding="utf-8")
    with pytest.raises(StylometryError, match="identifier column cannot"):
        read_feature_dataset(
            path,
            input_format="csv",
            selection=FeatureSelection(columns=("group", "f")),
        )


def test_reader_reports_missing_empty_and_invalid_utf8(tmp_path: Path) -> None:
    selection = FeatureSelection(prefixes=("f",))
    with pytest.raises(StylometryError, match="feature file not found"):
        read_feature_dataset(
            tmp_path / "missing.csv", input_format="csv", selection=selection
        )
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(StylometryError, match="no header"):
        read_feature_dataset(empty, input_format="csv", selection=selection)
    invalid = tmp_path / "invalid.csv"
    invalid.write_bytes(b"group,f\nA,1\nB,\xff\n")
    with pytest.raises(StylometryError, match="not valid UTF-8"):
        read_feature_dataset(invalid, input_format="csv", selection=selection)
