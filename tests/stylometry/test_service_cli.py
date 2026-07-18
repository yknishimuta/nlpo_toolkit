from __future__ import annotations

import io
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis import cli
from nlpo_toolkit.corpus_analysis.cli.stylometry_rendering import (
    write_burrows_delta_result,
)
from nlpo_toolkit.stylometry.models import (
    BurrowsDeltaRequest,
    FeatureDataset,
    FeatureObservation,
    FeatureSelection,
)
from nlpo_toolkit.stylometry.ports import StylometryCommandDependencies
from nlpo_toolkit.stylometry.results import BurrowsDeltaResult, DeltaPair
from nlpo_toolkit.stylometry.service import execute_burrows_delta


def test_service_reads_once_and_returns_feature_metadata(tmp_path: Path) -> None:
    calls = []
    dataset = FeatureDataset(
        ("varying", "constant"),
        (
            FeatureObservation("A", (1.0, 4.0)),
            FeatureObservation("B", (2.0, 4.0)),
        ),
    )

    def reader(path, *, input_format, selection):
        calls.append((path, input_format, selection))
        return dataset

    request = BurrowsDeltaRequest(
        tmp_path / "features.csv", "csv", FeatureSelection(prefixes=("f",))
    )
    result = execute_burrows_delta(
        request,
        dependencies=StylometryCommandDependencies(read_feature_dataset=reader),
    )

    assert calls == [(request.features_path, "csv", request.selection)]
    assert result.input_feature_names == ("varying", "constant")
    assert result.retained_feature_names == ("varying",)
    assert result.dropped_zero_variance_features == ("constant",)
    assert len(result.pairs) == 1


def test_renderer_preserves_result_order_and_supports_csv_tsv() -> None:
    result = BurrowsDeltaResult(
        pairs=(DeltaPair("B", "C", 2.0), DeltaPair("A", "B", 1.0)),
        input_feature_names=("f",),
        retained_feature_names=("f",),
        dropped_zero_variance_features=(),
    )
    csv_stream = io.StringIO()
    tsv_stream = io.StringIO()
    write_burrows_delta_result(result, stream=csv_stream, output_format="csv")
    write_burrows_delta_result(result, stream=tsv_stream, output_format="tsv")
    assert csv_stream.getvalue().splitlines() == [
        "sample_a,sample_b,burrows_delta",
        "B,C,2.0",
        "A,B,1.0",
    ]
    assert tsv_stream.getvalue().splitlines()[0] == (
        "sample_a\tsample_b\tburrows_delta"
    )


def test_cli_help_and_nested_command_requirement(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["stylometry", "delta", "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--features" in output
    assert "--feature-prefix" in output
    with pytest.raises(SystemExit) as missing:
        cli.main(["stylometry"])
    assert missing.value.code == 2


def test_cli_writes_stdout_and_reports_zero_variance_to_stderr(
    tmp_path: Path,
) -> None:
    path = tmp_path / "features.csv"
    path.write_text(
        "sample_id,mfw_a,mfw_constant\nA,1,4\nB,2,4\nC,3,4\n",
        encoding="utf-8",
    )
    stdout = io.StringIO()
    stderr = io.StringIO()
    assert (
        cli.main(
            [
                "stylometry",
                "delta",
                "--features",
                str(path),
                "--id-column",
                "sample_id",
                "--feature-prefix",
                "mfw_",
            ],
            stdout=stdout,
            stderr=stderr,
        )
        == 0
    )
    assert stdout.getvalue().splitlines()[0] == "sample_a,sample_b,burrows_delta"
    assert "excluded zero-variance features: 1" in stderr.getvalue()
    assert "STYLOMETRY" not in stdout.getvalue()


def test_cli_supports_tsv_input_output_and_output_path(tmp_path: Path) -> None:
    source = tmp_path / "features.tsv"
    output = tmp_path / "delta.tsv"
    source.write_text("group\tfw_et\nA\t1\nB\t2\n", encoding="utf-8")
    assert (
        cli.main(
            [
                "stylometry",
                "delta",
                "--features",
                str(source),
                "--input-format",
                "tsv",
                "--feature-column",
                "fw_et",
                "--format",
                "tsv",
                "--out",
                str(output),
            ],
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        == 0
    )
    assert output.read_text(encoding="utf-8").startswith(
        "sample_a\tsample_b\tburrows_delta\n"
    )


def test_cli_input_errors_return_one_without_traceback(tmp_path: Path) -> None:
    stderr = io.StringIO()
    assert (
        cli.main(
            [
                "stylometry",
                "delta",
                "--features",
                str(tmp_path / "missing.csv"),
                "--feature-prefix",
                "mfw_",
            ],
            stdout=io.StringIO(),
            stderr=stderr,
        )
        == 1
    )
    assert "feature file not found" in stderr.getvalue()
