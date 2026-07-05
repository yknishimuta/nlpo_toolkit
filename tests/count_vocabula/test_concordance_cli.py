from __future__ import annotations

import csv

from nlpo_toolkit.count_vocabula import cli
from nlpo_toolkit.count_vocabula.concordance import build_concordance_rows


def _write_trace(path):
    path.write_text(
        "\t".join(
            [
                "file",
                "group",
                "sentence",
                "token_idx",
                "token",
                "lemma",
                "upos",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "input/a.txt",
                "g1",
                "arma virumque cano Troiae qui primus ab oris",
                "1",
                "virumque",
                "vir",
                "NOUN",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "input/b.txt",
                "g2",
                "litora multum ille et terris iactatus et alto",
                "2",
                "ille",
                "ille",
                "PRON",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_concordance_writes_tsv_to_stdout(tmp_path, capsys):
    trace_path = tmp_path / "trace.tsv"
    _write_trace(trace_path)

    rc = cli.main(
        [
            "concordance",
            "--trace",
            str(trace_path),
            "--keys",
            "vir",
            "--field",
            "lemma",
            "--window",
            "1",
        ]
    )

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split("\t") == [
        "file",
        "group",
        "sentence",
        "key",
        "field",
        "token",
        "lemma",
        "left",
        "node",
        "right",
    ]
    row = lines[1].split("\t")
    assert row[0] == "input/a.txt"
    assert row[1] == "g1"
    assert row[3] == "vir"
    assert row[7] == "arma"
    assert row[8] == "virumque"
    assert row[9] == "cano"


def test_concordance_writes_csv_file_for_multiple_token_keys(tmp_path):
    trace_path = tmp_path / "trace.tsv"
    out_path = tmp_path / "kwic.csv"
    _write_trace(trace_path)

    rc = cli.main(
        [
            "concordance",
            "--trace",
            str(trace_path),
            "--keys",
            "virumque",
            "ille",
            "--field",
            "token",
            "--format",
            "csv",
            "--out",
            str(out_path),
        ]
    )

    assert rc == 0
    with out_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert [row["token"] for row in rows] == ["virumque", "ille"]
    assert rows[0]["field"] == "token"
    assert rows[1]["node"] == "ille"


def test_concordance_is_case_insensitive(tmp_path):
    trace_path = tmp_path / "trace.tsv"
    _write_trace(trace_path)

    columns, rows = build_concordance_rows(
        trace_path=trace_path,
        keys=["VIR"],
        field="lemma",
        window=2,
    )

    assert "sentence" in columns
    assert len(rows) == 1
    assert rows[0]["lemma"] == "vir"


def test_concordance_rejects_missing_trace(tmp_path, capsys):
    rc = cli.main(
        [
            "concordance",
            "--trace",
            str(tmp_path / "missing.tsv"),
            "--keys",
            "vir",
        ]
    )

    assert rc == 1
    assert "Trace not found" in capsys.readouterr().err


def test_concordance_rejects_negative_window(tmp_path, capsys):
    trace_path = tmp_path / "trace.tsv"
    _write_trace(trace_path)

    rc = cli.main(
        [
            "concordance",
            "--trace",
            str(trace_path),
            "--keys",
            "vir",
            "--window",
            "-1",
        ]
    )

    assert rc == 1
    assert "window must be zero or greater" in capsys.readouterr().err
