from __future__ import annotations

import csv
from collections import Counter

import pytest

from nlpo_toolkit.corpus_analysis import cli
from nlpo_toolkit.corpus_analysis.analysis_records import TokenRecord
from nlpo_toolkit.corpus_analysis.corpus import PreparedCorpus
from nlpo_toolkit.corpus_analysis.ngram import (
    ConfigSequenceToken, NgramError, NgramRow, build_ngrams_from_sequences,
    iter_config_sequence_tokens,
)
from nlpo_toolkit.corpus_analysis.token_artifact.paths import token_artifact_metadata_path
from nlpo_toolkit.corpus_analysis.token_artifact.schema import TokenArtifactDescriptor
from nlpo_toolkit.corpus_analysis.token_artifact.writer import TokenArtifactWriter
from nlpo_toolkit.corpus_analysis.token_sequences.grouping import build_token_sequence_collection


def record(token: str, global_index: int, *, group="g", file="a", section="s",
           chunk=0, sentence=0, index=0, lemma=None, included=True):
    return TokenRecord(group, file, chunk, sentence, index, global_index,
                       None, None, None, None, "metadata only", token,
                       lemma if lemma is not None else token, "NOUN", token,
                       included, None if included else "excluded", None,
                       section=section)


def rows(records, **kwargs):
    collection = build_token_sequence_collection(records)
    return build_ngrams_from_sequences(collection.sequences, **kwargs)


@pytest.mark.parametrize("changed", ["group", "file", "section", "chunk", "sentence"])
def test_bigram_does_not_cross_each_sequence_boundary(changed):
    values = dict(group="g", file="a", section="s", chunk=0, sentence=0)
    second = values | {changed: {"group":"h", "file":"b", "section":"t", "chunk":1, "sentence":1}[changed]}
    result = rows([
        record("a", 0, index=0, **values), record("b", 1, index=0, **second),
    ], n=2, field="token")
    assert result == []


def test_included_excluded_and_invalid_lexical_runs():
    result = rows([
        record("Arma", 0, index=0), record(",", 1, index=1, included=False),
        record("virumque", 2, index=2), record(".", 3, index=3),
        record("cano", 4, index=4), record("", 5, index=5, lemma=""),
        record("troiae", 6, index=6),
    ], n=2, field="token")
    assert result == [NgramRow("arma virumque", 1, 2, "token")]


def test_by_group_changes_only_aggregation_and_ordering():
    records = [
        record("a", 2, group="g2", index=0), record("b", 3, group="g2", index=1),
        record("A", 0, group="g1", index=0), record("b", 1, group="g1", index=1),
    ]
    all_rows = rows(records, n=2, field="token")
    grouped = rows(records, n=2, field="token", by_group=True)
    assert all_rows == [NgramRow("a b", 2, 2, "token")]
    assert grouped == [
        NgramRow("a b", 1, 2, "token", "g1"),
        NgramRow("a b", 1, 2, "token", "g2"),
    ]


def test_min_count_top_and_tie_sort():
    result = rows([
        record("b", 0, index=0), record("c", 1, index=1),
        record("a", 2, sentence=1, index=0), record("b", 3, sentence=1, index=1),
        record("a", 4, sentence=2, index=0), record("b", 5, sentence=2, index=1),
    ], n=2, field="token", min_count=2, top=1)
    assert result == [NgramRow("a b", 2, 2, "token")]


def test_config_adapter_uses_prepared_text_and_typed_sequences():
    corpora = (
        PreparedCorpus("g1", (), "raw", "quod vir", Counter()),
        PreparedCorpus("g2", (), "raw", "arma virum", Counter()),
    )
    tokens = list(iter_config_sequence_tokens(corpora))
    assert all(isinstance(token, ConfigSequenceToken) for token in tokens)
    assert [token.token for token in tokens] == ["quod", "vir", "arma", "virum"]
    assert [token.global_token_index for token in tokens] == [0, 1, 2, 3]
    assert rows(tokens, n=2, field="token") == [
        NgramRow("arma virum", 1, 2, "token"),
        NgramRow("quod vir", 1, 2, "token"),
    ]


def test_cli_reads_artifact_and_renders_existing_columns(tmp_path, capsys):
    path = tmp_path / "tokens.tsv"
    with TokenArtifactWriter(path, metadata_path=token_artifact_metadata_path(path),
                             descriptor=TokenArtifactDescriptor(group="g")) as writer:
        writer.write(record("Arma", 0, index=0))
        writer.write(record(",", 1, index=1, included=False))
        writer.write(record("virumque", 2, index=2))
    assert cli.main(["ngram", "--tokens", str(path), "--n", "2"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "ngram\tcount\tn\tfield", "arma virumque\t1\t2\tlemma",
    ]


def test_cli_csv_by_group(tmp_path):
    path, output = tmp_path / "tokens.tsv", tmp_path / "out.csv"
    with TokenArtifactWriter(path, metadata_path=token_artifact_metadata_path(path),
                             descriptor=TokenArtifactDescriptor(group="g")) as writer:
        writer.write(record("a", 0, index=0))
        writer.write(record("b", 1, index=1))
    assert cli.main(["ngram", "--tokens", str(path), "--n", "2", "--by-group",
                     "--format", "csv", "--out", str(output)]) == 0
    with output.open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream))[0]["group"] == "g"


def test_invalid_engine_arguments():
    with pytest.raises(NgramError):
        build_ngrams_from_sequences((), n=0, field="token")
