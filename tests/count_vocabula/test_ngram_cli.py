from __future__ import annotations

import csv

from nlpo_toolkit.count_vocabula import cli
from nlpo_toolkit.count_vocabula.ngram import build_ngrams_from_rows


def _write_trace(path):
    rows = [
        ["g1", "0", "Respondeo", "respondeo"],
        ["g1", "0", "dicendum", "dicendum"],
        ["g1", "0", "est", "sum"],
        ["g1", "0", ".", "."],
        ["g1", "1", "Sed", "sed"],
        ["g1", "1", "contra", "contra"],
        ["g1", "1", "est", "sum"],
        ["g2", "0", "Respondeo", "respondeo"],
        ["g2", "0", "dicendum", "dicendum"],
        ["g2", "0", "sit", "esse"],
    ]
    path.write_text(
        "group\tsent_idx\ttoken\tlemma\n"
        + "\n".join("\t".join(row) for row in rows)
        + "\n",
        encoding="utf-8",
    )


def test_build_lemma_bigrams_counts_rows():
    rows = [
        {"sent_idx": "0", "token": "Respondeo", "lemma": "respondeo"},
        {"sent_idx": "0", "token": "dicendum", "lemma": "dicendum"},
        {"sent_idx": "0", "token": "est", "lemma": "sum"},
        {"sent_idx": "1", "token": "Sed", "lemma": "sed"},
        {"sent_idx": "1", "token": "contra", "lemma": "contra"},
    ]

    out = build_ngrams_from_rows(rows, n=2, field="lemma")

    assert out == [
        {"ngram": "dicendum sum", "count": 1, "n": 2, "field": "lemma"},
        {"ngram": "respondeo dicendum", "count": 1, "n": 2, "field": "lemma"},
        {"ngram": "sed contra", "count": 1, "n": 2, "field": "lemma"},
    ]


def test_build_token_trigrams_do_not_cross_sent_idx():
    rows = [
        {"sent_idx": "0", "token": "a", "lemma": "a"},
        {"sent_idx": "0", "token": "b", "lemma": "b"},
        {"sent_idx": "1", "token": "c", "lemma": "c"},
        {"sent_idx": "1", "token": "d", "lemma": "d"},
        {"sent_idx": "1", "token": "e", "lemma": "e"},
    ]

    out = build_ngrams_from_rows(rows, n=3, field="token")

    assert out == [
        {"ngram": "c d e", "count": 1, "n": 3, "field": "token"},
    ]


def test_build_by_group_counts_separately():
    rows = [
        {"group": "g1", "sent_idx": "0", "token": "a", "lemma": "a"},
        {"group": "g1", "sent_idx": "0", "token": "b", "lemma": "b"},
        {"group": "g2", "sent_idx": "0", "token": "a", "lemma": "a"},
        {"group": "g2", "sent_idx": "0", "token": "b", "lemma": "b"},
    ]

    out = build_ngrams_from_rows(rows, n=2, field="lemma", by_group=True)

    assert out == [
        {"ngram": "a b", "count": 1, "n": 2, "field": "lemma", "group": "g1"},
        {"ngram": "a b", "count": 1, "n": 2, "field": "lemma", "group": "g2"},
    ]


def test_build_does_not_cross_group_boundary_when_aggregating_all_groups():
    rows = [
        {"group": "g1", "sent_idx": "0", "token": "a", "lemma": "a"},
        {"group": "g1", "sent_idx": "0", "token": "b", "lemma": "b"},
        {"group": "g2", "sent_idx": "0", "token": "c", "lemma": "c"},
        {"group": "g2", "sent_idx": "0", "token": "d", "lemma": "d"},
    ]

    out = build_ngrams_from_rows(rows, n=2, field="lemma")

    assert out == [
        {"ngram": "a b", "count": 1, "n": 2, "field": "lemma"},
        {"ngram": "c d", "count": 1, "n": 2, "field": "lemma"},
    ]


def test_build_filters_symbol_only_tokens():
    rows = [
        {"sent_idx": "0", "token": "a", "lemma": "a"},
        {"sent_idx": "0", "token": ".", "lemma": "."},
        {"sent_idx": "0", "token": "b", "lemma": "b"},
        {"sent_idx": "0", "token": "c", "lemma": "c"},
    ]

    out = build_ngrams_from_rows(rows, n=2, field="token")

    assert out == [
        {"ngram": "b c", "count": 1, "n": 2, "field": "token"},
    ]


def test_ngram_cli_writes_tsv_stdout_with_min_count_and_top(tmp_path, capsys):
    trace_path = tmp_path / "trace.tsv"
    _write_trace(trace_path)

    rc = cli.main(
        [
            "ngram",
            "--trace",
            str(trace_path),
            "--n",
            "2",
            "--field",
            "lemma",
            "--min-count",
            "2",
            "--top",
            "1",
        ]
    )

    assert rc == 0
    assert capsys.readouterr().out.splitlines() == [
        "ngram\tcount\tn\tfield",
        "respondeo dicendum\t2\t2\tlemma",
    ]


def test_ngram_cli_writes_csv_by_group(tmp_path):
    trace_path = tmp_path / "trace.tsv"
    out_path = tmp_path / "ngrams.csv"
    _write_trace(trace_path)

    rc = cli.main(
        [
            "ngram",
            "--trace",
            str(trace_path),
            "--n",
            "3",
            "--field",
            "token",
            "--by-group",
            "--format",
            "csv",
            "--out",
            str(out_path),
        ]
    )

    assert rc == 0
    with out_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0] == {
        "ngram": "respondeo dicendum est",
        "count": "1",
        "n": "3",
        "field": "token",
        "group": "g1",
    }
    assert {row["group"] for row in rows} == {"g1", "g2"}


def test_ngram_cli_rejects_missing_trace_field(tmp_path, capsys):
    trace_path = tmp_path / "trace.tsv"
    trace_path.write_text("token\narma\n", encoding="utf-8")

    rc = cli.main(["ngram", "--trace", str(trace_path), "--field", "lemma"])

    assert rc == 1
    assert "Trace must contain 'lemma' column" in capsys.readouterr().err


def test_ngram_cli_rejects_missing_group_column_for_by_group(tmp_path, capsys):
    trace_path = tmp_path / "trace.tsv"
    trace_path.write_text("sent_idx\ttoken\tlemma\n0\tarma\tarma\n", encoding="utf-8")

    rc = cli.main(["ngram", "--trace", str(trace_path), "--by-group"])

    assert rc == 1
    assert "Trace must contain 'group' column when --by-group is used" in capsys.readouterr().err
