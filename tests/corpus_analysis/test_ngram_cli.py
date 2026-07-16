from __future__ import annotations

import csv
from collections import Counter

import pytest

from nlpo_toolkit.corpus_analysis import cli
from nlpo_toolkit.corpus_analysis.config import load_config
from nlpo_toolkit.corpus_analysis.analysis_records import TokenRecord
from nlpo_toolkit.corpus_analysis.corpus import PreparedCorpus
from nlpo_toolkit.corpus_analysis.ports import (
    ConfigNgramDependencies,
    CorpusExecutionDependencies,
    CorpusPlanningDependencies,
    CorpusPreparationDependencies,
)
from nlpo_toolkit.corpus_analysis.ngram import (
    ConfigNgramRequest,
    build_ngrams_from_rows,
    iter_config_token_rows,
    read_token_artifact_rows,
    execute_config_ngram_command,
)
from nlpo_toolkit.corpus_analysis.requests import CorpusPreparationRequest
from nlpo_toolkit.latin.cleaners.config_loader import inspect_cleaner_config
from nlpo_toolkit.corpus_analysis.token_artifact import (
    TokenArtifactMetadata,
    TokenArtifactWriter,
    token_artifact_metadata_path,
)


def _write_artifact(path):
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
    records = []
    global_index = 0
    sentence_positions: dict[tuple[str, str], int] = {}
    for group, sentence_index, token, lemma in rows:
        key = (group, sentence_index)
        token_index = sentence_positions.get(key, 0)
        sentence_positions[key] = token_index + 1
        records.append(
            TokenRecord(group, f"input/{group}.txt", 0, int(sentence_index), token_index, global_index, None, None, None, None, "", token, lemma, "NOUN", lemma, True, None, None)
        )
        global_index += 1
    with TokenArtifactWriter(path, token_artifact_metadata_path(path), metadata=TokenArtifactMetadata(group="mixed")) as writer:
        for record in records:
            writer.write(record)


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
    tokens_path = tmp_path / "tokens.tsv"
    _write_artifact(tokens_path)

    rc = cli.main(
        [
            "ngram",
            "--tokens",
            str(tokens_path),
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
    tokens_path = tmp_path / "tokens.tsv"
    out_path = tmp_path / "ngrams.csv"
    _write_artifact(tokens_path)

    rc = cli.main(
        [
            "ngram",
            "--tokens",
            str(tokens_path),
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


def test_ngram_cli_reads_token_artifact_and_respects_boundaries(tmp_path, capsys):
    artifact = tmp_path / "tokens.tsv"
    records = [
        TokenRecord("g1", "input/a.txt", 0, 0, 0, 0, None, None, None, None, "a b", "a", "a", "NOUN", "a", True, None, None),
        TokenRecord("g1", "input/a.txt", 0, 0, 1, 1, None, None, None, None, "a b", "b", "b", "NOUN", "b", True, None, None),
        TokenRecord("g1", "input/a.txt", 0, 1, 0, 2, None, None, None, None, "c d", "c", "c", "NOUN", "c", True, None, None),
        TokenRecord("g1", "input/b.txt", 0, 1, 1, 3, None, None, None, None, "c d", "d", "d", "NOUN", "d", True, None, None),
        TokenRecord("g2", "input/b.txt", 0, 0, 0, 4, None, None, None, None, "e f", "e", "e", "NOUN", "e", True, None, None),
        TokenRecord("g2", "input/b.txt", 0, 0, 1, 5, None, None, None, None, "e f", "f", "f", "NOUN", "f", True, None, None),
    ]
    with TokenArtifactWriter(artifact, token_artifact_metadata_path(artifact), metadata=TokenArtifactMetadata(group="mixed")) as writer:
        for record in records:
            writer.write(record)

    rc = cli.main(["ngram", "--tokens", str(artifact), "--n", "2", "--field", "lemma"])

    assert rc == 0
    assert capsys.readouterr().out.splitlines() == [
        "ngram\tcount\tn\tfield",
        "a b\t1\t2\tlemma",
        "e f\t1\t2\tlemma",
    ]


def test_artifact_ngram_uses_included_records_order_and_all_boundaries(tmp_path):
    artifact = tmp_path / "tokens.tsv"
    records = [
        TokenRecord("g", "a.txt", 0, 0, 1, 2, None, None, None, None, "", "b", "b", "NOUN", "b", True, None, None, section="s1"),
        TokenRecord("g", "a.txt", 0, 0, 0, 1, None, None, None, None, "", "a", "a", "NOUN", "a", True, None, None, section="s1"),
        TokenRecord("g", "a.txt", 0, 0, 2, 3, None, None, None, None, "", "x", "x", "PUNCT", None, False, "upos_not_targeted", None, section="s1"),
        TokenRecord("g", "a.txt", 0, 0, 0, 4, None, None, None, None, "", "c", "c", "NOUN", "c", True, None, None, section="s2"),
        TokenRecord("g", "a.txt", 1, 0, 0, 5, None, None, None, None, "", "d", "d", "NOUN", "d", True, None, None, section="s2"),
        TokenRecord("g", "b.txt", 1, 0, 0, 6, None, None, None, None, "", "e", "e", "NOUN", "e", True, None, None, section="s2"),
        TokenRecord("h", "b.txt", 1, 0, 0, 7, None, None, None, None, "", "f", "f", "NOUN", "f", True, None, None, section="s2"),
    ]
    with TokenArtifactWriter(artifact, token_artifact_metadata_path(artifact), metadata=TokenArtifactMetadata(group="mixed")) as writer:
        for record in records:
            writer.write(record)

    rows = read_token_artifact_rows(artifact, "lemma")
    result = build_ngrams_from_rows(rows, n=2, field="lemma")

    assert [row["lemma"] for row in rows] == ["a", "b", "c", "d", "e", "f"]
    assert result == [{"ngram": "a b", "count": 1, "n": 2, "field": "lemma"}]


def test_ngram_cli_rejects_tsv_without_artifact_metadata(tmp_path, capsys):
    trace_path = tmp_path / "trace.tsv"
    trace_path.write_text("token\narma\n", encoding="utf-8")

    rc = cli.main(["ngram", "--tokens", str(trace_path), "--field", "lemma"])

    assert rc == 1
    assert "metadata" in capsys.readouterr().err


def test_iter_config_token_rows_uses_prepared_text_and_preserves_group_boundary():
    corpora = (
        PreparedCorpus("g1", (), "uod TAG vir", "quod vir", Counter()),
        PreparedCorpus("g2", (), "raw", "arma virum", Counter()),
    )

    rows = list(iter_config_token_rows(corpora))

    assert rows == [
        {"group": "g1", "token": "quod"},
        {"group": "g1", "token": "vir"},
        {"group": "g2", "token": "arma"},
        {"group": "g2", "token": "virum"},
    ]
    assert build_ngrams_from_rows(rows, n=2, field="token") == [
        {"ngram": "arma virum", "count": 1, "n": 2, "field": "token"},
        {"ngram": "quod vir", "count": 1, "n": 2, "field": "token"},
    ]


def test_config_ngram_uses_canonical_analysis_plan_with_overrides(tmp_path, monkeypatch):
    import nlpo_toolkit.corpus_analysis.ngram as ngram_mod

    config_path = tmp_path / "groups.yml"
    config_path.write_text("groups:\n  text:\n    files: [input.txt]\n", encoding="utf-8")
    calls = []

    def fake_prepare_session(request, **kwargs):
        calls.append((request, kwargs))
        return type(
            "Session",
            (),
            {
                "corpora": (
                    PreparedCorpus("text", (), "raw", "alpha beta", Counter()),
                ),
            },
        )()

    monkeypatch.setattr(ngram_mod, "prepare_analysis_corpus_session", fake_prepare_session)

    result = execute_config_ngram_command(
        request=ConfigNgramRequest(
            corpus=CorpusPreparationRequest(
                tmp_path,
                config_path,
                grouping_override="auto_single_cleaned",
                error_on_empty_group=True,
            ),
            n=2,
            by_group=True,
            min_count=1,
            top=None,
        ),
        dependencies=ConfigNgramDependencies(
            corpus=CorpusExecutionDependencies(
                planning=CorpusPlanningDependencies(
                    load_config=load_config,
                    cleaner_inspector=inspect_cleaner_config,
                ),
                preparation=CorpusPreparationDependencies(
                    execute_cleaner=lambda _request: pytest.fail("cleaner service must not be called")
                ),
            ),
        ),
    )

    assert result.rows == (
        {"ngram": "alpha beta", "count": 1, "n": 2, "field": "token", "group": "text"},
    )
    assert len(calls) == 1
    assert calls[0][0] is not None
    assert calls[0][0].project_root == tmp_path
    assert calls[0][0].config_path == config_path
    assert calls[0][0].grouping_override == "auto_single_cleaned"
    assert calls[0][0].error_on_empty_group is True
    assert set(calls[0][1]) == {"dependencies"}


def test_config_ngram_does_not_apply_count_partition_validation(tmp_path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("alpha beta", encoding="utf-8")
    config_path = tmp_path / "groups.yml"
    config_path.write_text(
        "groups:\n"
        "  whole: {files: [input/a.txt]}\n"
        "  part_a: {files: [input/a.txt]}\n"
        "  part_b: {files: [input/a.txt]}\n"
        "validations:\n"
        "  partitions:\n"
        "    - {name: split, whole: whole, parts: [part_a, part_b]}\n",
        encoding="utf-8",
    )

    result = execute_config_ngram_command(
        request=ConfigNgramRequest(
            corpus=CorpusPreparationRequest(
                tmp_path, config_path, grouping_override="per_file"
            ),
            n=2,
            by_group=False,
            min_count=1,
            top=None,
        ),
        dependencies=ConfigNgramDependencies(
            corpus=CorpusExecutionDependencies(
                planning=CorpusPlanningDependencies(
                    load_config=load_config,
                    cleaner_inspector=inspect_cleaner_config,
                ),
                preparation=CorpusPreparationDependencies(
                    execute_cleaner=lambda _request: pytest.fail("cleaner must not run")
                ),
            ),
        ),
    )

    assert result.rows == (
        {"ngram": "alpha beta", "count": 1, "n": 2, "field": "token"},
    )


def test_token_artifact_cli_does_not_create_config_dependencies(tmp_path, monkeypatch, capsys):
    import nlpo_toolkit.corpus_analysis.cli.ngram as ngram_cli

    artifact = tmp_path / "tokens.tsv"
    _write_artifact(artifact)
    monkeypatch.setattr(
        ngram_cli,
        "default_config_ngram_dependencies",
        lambda: pytest.fail("config dependencies must not be created"),
    )
    monkeypatch.setattr(
        ngram_cli,
        "build_corpus_preparation_request",
        lambda _args: pytest.fail("corpus request must not be created"),
    )

    rc = cli.main(["ngram", "--tokens", str(artifact), "--field", "token", "--n", "2"])

    assert rc == 0
    assert capsys.readouterr().out.splitlines()[0] == "ngram\tcount\tn\tfield"


def test_config_ngram_cli_always_uses_token_field(tmp_path, capsys) -> None:
    (tmp_path / "input.txt").write_text("Alpha beta", encoding="utf-8")
    config_path = tmp_path / "groups.yml"
    config_path.write_text(
        "groups:\n  text: {files: [input.txt]}\n",
        encoding="utf-8",
    )

    rc = cli.main(
        [
            "ngram",
            "--project-root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--field",
            "lemma",
            "--n",
            "2",
        ]
    )

    assert rc == 0
    assert capsys.readouterr().out.splitlines() == [
        "ngram\tcount\tn\tfield",
        "alpha beta\t1\t2\ttoken",
    ]
