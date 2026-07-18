from __future__ import annotations

import io
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis import cli
from nlpo_toolkit.corpus_analysis.cli.stylometry_neighbor_rendering import (
    NEIGHBOR_COLUMNS,
    write_neighbor_result,
)
from nlpo_toolkit.stylometry.metrics import StylometryMetric
from nlpo_toolkit.stylometry.models import (
    FeatureDataset,
    FeatureObservation,
    FeatureSelection,
)
from nlpo_toolkit.stylometry.neighbor_models import NeighborRankingRequest
from nlpo_toolkit.stylometry.neighbor_service import execute_neighbor_ranking
from nlpo_toolkit.stylometry.ports import StylometryCommandDependencies


def test_service_reads_once_and_uses_existing_standardization(tmp_path: Path) -> None:
    calls = []
    dataset = FeatureDataset(
        ("varying", "constant"),
        (
            FeatureObservation("A", (1.0, 4.0)),
            FeatureObservation("B", (2.0, 4.0)),
            FeatureObservation("C", (3.0, 4.0)),
        ),
    )

    def reader(path, *, input_format, selection):
        calls.append((path, input_format, selection))
        return dataset

    request = NeighborRankingRequest(
        tmp_path / "features.csv",
        "csv",
        FeatureSelection(prefixes=("f",)),
        StylometryMetric.MANHATTAN,
        1,
    )
    result = execute_neighbor_ranking(
        request,
        dependencies=StylometryCommandDependencies(
            read_feature_dataset=reader,
            read_authorship_metadata=lambda *args, **kwargs: pytest.fail(
                "metadata reader must not be called"
            ),
        ),
    )
    assert calls == [(request.features_path, "csv", request.selection)]
    assert result.input_feature_names == ("varying", "constant")
    assert result.retained_feature_names == ("varying",)
    assert result.dropped_zero_variance_features == ("constant",)
    assert result.query_count == result.output_row_count == 3


def test_renderer_schema_order_no_rounding_and_tsv(tmp_path: Path) -> None:
    dataset = FeatureDataset(
        ("f",),
        (FeatureObservation("A", (1.0,)), FeatureObservation("B", (2.0,))),
    )
    request = NeighborRankingRequest(
        tmp_path / "x", "csv", FeatureSelection(columns=("f",))
    )
    result = execute_neighbor_ranking(
        request,
        dependencies=StylometryCommandDependencies(
            read_feature_dataset=lambda *args, **kwargs: dataset,
            read_authorship_metadata=lambda *args, **kwargs: pytest.fail("unused"),
        ),
    )
    stream = io.StringIO()
    write_neighbor_result(result, stream=stream, output_format="tsv")
    assert stream.getvalue().splitlines()[0] == "\t".join(NEIGHBOR_COLUMNS)
    assert stream.getvalue().splitlines()[1].startswith("A\t1\tB\tburrows_delta\t")


def test_cli_help_stdout_metric_top_and_zero_variance(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["stylometry", "neighbors", "--help"])
    assert exc.value.code == 0
    assert "--metric" in capsys.readouterr().out
    path = tmp_path / "features.csv"
    path.write_text(
        "sample_id,mfw_a,mfw_constant\nA,1,4\nB,2,4\nC,3,4\n",
        encoding="utf-8",
    )
    stdout, stderr = io.StringIO(), io.StringIO()
    assert (
        cli.main(
            [
                "stylometry",
                "neighbors",
                "--features",
                str(path),
                "--id-column",
                "sample_id",
                "--feature-prefix",
                "mfw_",
                "--metric",
                "manhattan",
                "--top",
                "1",
            ],
            stdout=stdout,
            stderr=stderr,
        )
        == 0
    )
    lines = stdout.getvalue().splitlines()
    assert lines[0] == ",".join(NEIGHBOR_COLUMNS)
    assert len(lines) == 4
    assert all(",manhattan," in line for line in lines[1:])
    assert "excluded zero-variance features: 1" in stderr.getvalue()


def test_cli_tsv_output_and_cosine_zero_norm_error(tmp_path: Path) -> None:
    path = tmp_path / "features.tsv"
    output = tmp_path / "neighbors.tsv"
    path.write_text("group\tf1\tf2\nA\t0\t1\nB\t0\t2\nC\t0\t3\n", encoding="utf-8")
    stderr = io.StringIO()
    code = cli.main(
        [
            "stylometry",
            "neighbors",
            "--features",
            str(path),
            "--input-format",
            "tsv",
            "--feature-column",
            "f1",
            "--feature-column",
            "f2",
            "--metric",
            "cosine_similarity",
            "--format",
            "tsv",
            "--out",
            str(output),
        ],
        stdout=io.StringIO(),
        stderr=stderr,
    )
    assert code == 1
    assert "zero norm" in stderr.getvalue()
    assert not output.exists()
