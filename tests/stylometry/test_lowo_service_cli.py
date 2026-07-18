from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis import cli
from nlpo_toolkit.corpus_analysis.cli.stylometry_evaluation_rendering import (
    FOLD_COLUMNS,
    summary_json_value,
)
from nlpo_toolkit.stylometry.evaluation_models import (
    AuthorshipAssignment,
    AuthorshipMetadata,
    LeaveOneWorkOutEvaluationRequest,
)
from nlpo_toolkit.stylometry.evaluation_service import execute_lowo_evaluation
from nlpo_toolkit.stylometry.models import (
    FeatureDataset,
    FeatureObservation,
    FeatureSelection,
)
from nlpo_toolkit.stylometry.ports import StylometryCommandDependencies


def _dataset() -> FeatureDataset:
    return FeatureDataset(
        ("fw_et",),
        tuple(
            FeatureObservation(identifier, (value,))
            for identifier, value in (
                ("A1", 0.0),
                ("A2", 1.0),
                ("B1", 9.0),
                ("B2", 11.0),
            )
        ),
    )


def _metadata() -> AuthorshipMetadata:
    return AuthorshipMetadata(
        tuple(
            AuthorshipAssignment(identifier, author, identifier)
            for identifier, author in (
                ("A1", "A"),
                ("A2", "A"),
                ("B1", "B"),
                ("B2", "B"),
            )
        )
    )


def test_service_calls_each_reader_once_and_returns_typed_result(
    tmp_path: Path,
) -> None:
    feature_calls = []
    metadata_calls = []

    def feature_reader(path, *, input_format, selection):
        feature_calls.append((path, input_format, selection))
        return _dataset()

    def metadata_reader(path, **kwargs):
        metadata_calls.append((path, kwargs))
        return _metadata()

    request = LeaveOneWorkOutEvaluationRequest(
        features_path=tmp_path / "features.csv",
        input_format="csv",
        feature_selection=FeatureSelection(columns=("fw_et",)),
        metadata_path=tmp_path / "metadata.csv",
        metadata_format="csv",
        metadata_id_column="sample_id",
        author_column="author",
        work_column="work",
    )
    result = execute_lowo_evaluation(
        request,
        dependencies=StylometryCommandDependencies(feature_reader, metadata_reader),
    )
    assert len(feature_calls) == len(metadata_calls) == 1
    assert result.summary.accuracy == 1.0
    assert len(result.folds) == 4


def test_summary_json_schema_is_strict_and_complete(tmp_path: Path) -> None:
    request = LeaveOneWorkOutEvaluationRequest(
        tmp_path / "features.csv",
        "csv",
        FeatureSelection(columns=("fw_et",)),
        tmp_path / "metadata.csv",
        "csv",
        "id",
        "author",
        "work",
    )
    result = execute_lowo_evaluation(
        request,
        dependencies=StylometryCommandDependencies(
            lambda *args, **kwargs: _dataset(),
            lambda *args, **kwargs: _metadata(),
        ),
    )
    value = summary_json_value(result)
    encoded = json.dumps(value, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["method"] == "leave_one_work_out"
    assert decoded["classifier"] == "burrows_delta_author_centroid"
    assert decoded["work_count"] == 4
    assert decoded["accuracy"] == 1.0
    assert [item["author"] for item in decoded["authors"]] == ["A", "B"]


def test_cli_evaluate_lowo_outputs_folds_summary_and_stderr(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    metadata = tmp_path / "metadata.csv"
    summary = tmp_path / "summary.json"
    features.write_text("sample_id,fw_et\nA1,0\nA2,1\nB1,9\nB2,11\n", encoding="utf-8")
    metadata.write_text(
        "sample_id,author,work\nA1,A,A1\nA2,A,A2\nB1,B,B1\nB2,B,B2\n",
        encoding="utf-8",
    )
    stdout = io.StringIO()
    stderr = io.StringIO()
    assert (
        cli.main(
            [
                "stylometry",
                "evaluate-lowo",
                "--features",
                str(features),
                "--metadata",
                str(metadata),
                "--id-column",
                "sample_id",
                "--feature-prefix",
                "fw_",
                "--summary-out",
                str(summary),
            ],
            stdout=stdout,
            stderr=stderr,
        )
        == 0
    )
    assert tuple(stdout.getvalue().splitlines()[0].split(",")) == FOLD_COLUMNS
    assert len(stdout.getvalue().splitlines()) == 5
    assert "LOWO work accuracy: 4/4 (1.0)" in stderr.getvalue()
    assert json.loads(summary.read_text(encoding="utf-8"))["accuracy"] == 1.0


def test_cli_help_defaults_and_same_output_error(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["stylometry", "evaluate-lowo", "--help"])
    assert exc.value.code == 0
    assert "--metadata-id-column" in capsys.readouterr().out
    path = tmp_path / "same.csv"
    stderr = io.StringIO()
    assert (
        cli.main(
            [
                "stylometry",
                "evaluate-lowo",
                "--features",
                "features.csv",
                "--metadata",
                "metadata.csv",
                "--feature-prefix",
                "fw_",
                "--out",
                str(path),
                "--summary-out",
                str(path),
            ],
            stdout=io.StringIO(),
            stderr=stderr,
        )
        == 1
    )
    assert "must be different paths" in stderr.getvalue()
    assert not path.exists()
