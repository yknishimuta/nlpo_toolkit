from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis import cli
from nlpo_toolkit.corpus_analysis.cli.stylometry_verification_rendering import (
    CALIBRATION_COLUMNS,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    features = tmp_path / "features.csv"
    metadata = tmp_path / "metadata.csv"
    features.write_text(
        "sample_id,fw_et\n"
        "a1,0\na2,1\na3,2\nb1,8\nb2,10\nq1,1\nq2,3\n",
        encoding="utf-8",
    )
    metadata.write_text(
        "sample_id,author,work\n"
        "a1,A,A1\na2,A,A2\na3,A,A3\n"
        "b1,B,B1\nb2,C,B2\nq1,unknown,Q\nq2,unknown,Q\n",
        encoding="utf-8",
    )
    return features, metadata


def test_verify_cli_writes_strict_json_and_calibration(tmp_path: Path) -> None:
    features, metadata = _inputs(tmp_path)
    calibration = tmp_path / "calibration.tsv"
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = cli.main(
        [
            "stylometry", "verify", "--features", str(features),
            "--metadata", str(metadata), "--id-column", "sample_id",
            "--feature-prefix", "fw_", "--candidate-author", "A",
            "--query-work", "Q", "--calibration-out", str(calibration),
            "--calibration-format", "tsv",
        ],
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0
    value = json.loads(stdout.getvalue())
    assert value["method"] == "candidate_authorship_verification"
    assert value["decision"] in {"accept", "reject", "inconclusive"}
    assert value["query_sample_count"] == 2
    assert value["limitations"]["authenticity_not_proven"] is True
    with calibration.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    assert tuple(rows[0]) == CALIBRATION_COLUMNS
    assert [row[0] for row in rows[1:]] == ["genuine"] * 3 + ["impostor"] * 2
    assert stderr.getvalue() == ""


def test_verify_cli_file_output_help_and_same_path_guard(
    tmp_path: Path, capsys
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["stylometry", "verify", "--help"])
    assert exc.value.code == 0
    assert "--candidate-author" in capsys.readouterr().out
    features, metadata = _inputs(tmp_path)
    output = tmp_path / "verification.json"
    assert cli.main(
        [
            "stylometry", "verify", "--features", str(features),
            "--metadata", str(metadata), "--id-column", "sample_id",
            "--feature-prefix", "fw_", "--candidate-author", "A",
            "--query-work", "Q", "--out", str(output),
        ],
        stdout=io.StringIO(), stderr=io.StringIO(),
    ) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["query_work"] == "Q"

    same = tmp_path / "same.json"
    stderr = io.StringIO()
    assert cli.main(
        [
            "stylometry", "verify", "--features", str(features),
            "--metadata", str(metadata), "--feature-prefix", "fw_",
            "--candidate-author", "A", "--query-work", "Q",
            "--out", str(same), "--calibration-out", str(same),
        ],
        stdout=io.StringIO(), stderr=stderr,
    ) == 1
    assert "must be different paths" in stderr.getvalue()
    assert not same.exists()


def test_verify_corpus_help_and_output_collision(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["stylometry", "verify-corpus", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--candidate-author" in help_text
    assert "--query-work" in help_text
    assert "--vocabulary-audit-out" in help_text

    same = tmp_path / "same.json"
    stderr = io.StringIO()
    assert (
        cli.main(
            [
                "stylometry",
                "verify-corpus",
                "--metadata",
                str(tmp_path / "metadata.csv"),
                "--candidate-author",
                "A",
                "--query-work",
                "Q",
                "--out",
                str(same),
                "--vocabulary-audit-out",
                str(same),
            ],
            stdout=io.StringIO(),
            stderr=stderr,
        )
        == 1
    )
    assert "outputs must differ" in stderr.getvalue()
    assert not same.exists()
