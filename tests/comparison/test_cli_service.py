from __future__ import annotations

import csv
import io
import math
from pathlib import Path

from nlpo_toolkit.comparison.cli_service import (
    CompareError,
    CompareCommandResult,
    CompareRequest,
    compare_frequency_tables,
    execute_compare_command,
)
from nlpo_toolkit.comparison.frequency_io import detect_columns, load_frequency_csv
from nlpo_toolkit.corpus_analysis.cli.output import write_compare_result


def run_compare(**kwargs):
    out = kwargs.pop("out", None)
    output_format = kwargs.pop("output_format", "csv")
    result = execute_compare_command(
        CompareRequest(
            inputs=tuple(kwargs.pop("inputs")),
            labels=(
                tuple(kwargs.pop("labels"))
                if kwargs.get("labels") is not None
                else kwargs.pop("labels", None)
            ),
            **kwargs,
        )
    )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as stream:
            write_compare_result(result, stream=stream, output_format=output_format)
    return 0


def write_compare_output(rows, *, out, format):
    result = CompareCommandResult(
        rows=tuple(rows),
        columns=tuple(rows[0]) if rows else ("term",),
    )
    write_compare_result(result, stream=out, output_format=format)


def _write_csv(path: Path, header: str, rows: list[str]) -> Path:
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_detect_columns_auto_and_explicit(tmp_path: Path) -> None:
    assert detect_columns(["lemma", "count"]) == ("lemma", "count")
    assert detect_columns(["token", "frequency"]) == ("token", "frequency")
    assert detect_columns(["word", "n"], key_column="word", count_column="n") == ("word", "n")


def test_load_frequency_csv_auto_detects_and_ignores_empty_keys(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "freq.csv",
        "lemma,count",
        ["rosa,2", ",10", "arma,3"],
    )

    assert load_frequency_csv(path) == {"rosa": 2.0, "arma": 3.0}


def test_load_frequency_csv_explicit_columns(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "freq.csv", "word,n", ["rosa,2"])

    assert load_frequency_csv(path, key_column="word", count_column="n") == {"rosa": 2.0}


def test_compare_two_inputs_includes_missing_terms_and_metrics() -> None:
    rows = compare_frequency_tables(
        [{"rosa": 2.0, "arma": 1.0}, {"rosa": 1.0, "vir": 3.0}],
        ["a", "b"],
        smoothing=0.5,
    )
    by_term = {row["term"]: row for row in rows}

    assert by_term["arma"]["a_count"] == 1.0
    assert by_term["arma"]["b_count"] == 0.0
    assert by_term["vir"]["a_count"] == 0.0
    assert by_term["vir"]["b_count"] == 3.0
    assert by_term["rosa"]["difference"] == by_term["rosa"]["a_relative"] - by_term["rosa"]["b_relative"]
    assert by_term["arma"]["ratio"] > 1
    assert math.isclose(by_term["arma"]["log_ratio"], math.log2(by_term["arma"]["ratio"]))


def test_compare_metrics_are_available() -> None:
    rows = compare_frequency_tables(
        [{"a": 2.0, "b": 1.0}, {"a": 1.0, "b": 3.0}],
        ["left", "right"],
        smoothing=0.5,
    )
    row = {r["term"]: r for r in rows}["a"]

    assert row["left_relative"] > row["right_relative"]
    assert row["difference"] > 0
    assert row["ratio"] > 1
    assert row["log_ratio"] > 0


def test_min_total_count_filters_terms() -> None:
    rows = compare_frequency_tables(
        [{"rare": 1.0, "common": 3.0}, {"common": 2.0}],
        ["a", "b"],
        min_total_count=2,
    )

    assert {row["term"] for row in rows} == {"common"}


def test_run_compare_top_and_output_file(tmp_path: Path) -> None:
    a = _write_csv(tmp_path / "a.csv", "lemma,count", ["x,10", "y,1"])
    b = _write_csv(tmp_path / "b.csv", "lemma,count", ["x,1", "y,10"])
    out = tmp_path / "compare.csv"

    rc = run_compare(
        inputs=[a, b],
        labels=["a", "b"],
        out=out,
        top=1,
        sort="term",
        ascending=True,
    )

    assert rc == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["term"] == "x"


def test_run_compare_accepts_new_and_legacy_frequency_names(tmp_path: Path) -> None:
    new_name = _write_csv(tmp_path / "frequency_text.csv", "lemma,count", ["x,2"])
    legacy_name = _write_csv(tmp_path / "noun_frequency_text.csv", "lemma,count", ["x,1"])
    out = tmp_path / "compare.csv"

    rc = run_compare(inputs=[new_name, legacy_name], labels=["new", "legacy"], out=out)

    assert rc == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0]["new_count"] == "2"
    assert rows[0]["legacy_count"] == "1"


def test_three_input_compare_has_range_columns() -> None:
    rows = compare_frequency_tables(
        [{"rosa": 10.0, "arma": 1.0}, {"rosa": 1.0, "arma": 10.0}, {"rosa": 5.0, "arma": 5.0}],
        ["a", "b", "c"],
        smoothing=0.5,
    )
    row = {r["term"]: r for r in rows}["rosa"]

    assert row["max_label"] == "a"
    assert row["min_label"] == "b"
    assert row["range_relative"] == row["max_relative"] - row["min_relative"]


def test_write_compare_output_csv_and_tsv() -> None:
    rows = [{"term": "rosa", "a_count": 1.0, "total_count": 1.0}]
    csv_out = io.StringIO()
    tsv_out = io.StringIO()

    write_compare_output(rows, out=csv_out, format="csv")
    write_compare_output(rows, out=tsv_out, format="tsv")

    assert "term,a_count,total_count" in csv_out.getvalue()
    assert "term\ta_count\ttotal_count" in tsv_out.getvalue()


def test_run_compare_label_mismatch_returns_error(tmp_path: Path) -> None:
    a = _write_csv(tmp_path / "a.csv", "lemma,count", ["x,1"])
    b = _write_csv(tmp_path / "b.csv", "lemma,count", ["x,1"])

    try:
        run_compare(inputs=[a, b], labels=["only-one"])
    except CompareError as exc:
        assert "--labels" in str(exc)
    else:
        raise AssertionError("expected CompareError")


def test_load_frequency_csv_invalid_count_errors(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "freq.csv", "lemma,count", ["rosa,nope"])

    try:
        load_frequency_csv(path)
    except CompareError as exc:
        assert "Invalid numeric count" in str(exc)
    else:
        raise AssertionError("expected CompareError")
