from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis import cli
from nlpo_toolkit.corpus_analysis.features import (
    FeatureError,
    FeatureOptions,
    TokenRecord,
    build_feature_rows,
    compute_basic_features,
    compute_upos_features,
    run_features,
    safe_feature_name,
    write_feature_matrix,
)
from nlpo_toolkit.models import NLPDocument, NLPSentence, NLPToken
from nlpo_toolkit.backends import BuiltNLPBackend, NLPBackendInfo
from nlpo_toolkit.corpus_analysis.config import NLPConfig


def _doc_for_text(text: str) -> NLPDocument:
    words = {
        "rosa": NLPToken("Rosa", "rosa", "NOUN"),
        "amat": NLPToken("amat", "amo", "VERB"),
        "et": NLPToken("et", "et", "CCONJ"),
        "puella": NLPToken("puella", "puella", "NOUN"),
        "in": NLPToken("in", "in", "ADP"),
        "villa": NLPToken("villa", "villa", "NOUN"),
        "currit": NLPToken("currit", "curro", "VERB"),
        ".": NLPToken(".", ".", "PUNCT"),
    }
    tokens = [words[w] for w in text.lower().replace(".", " .").split() if w in words]
    sentences = [NLPSentence(tokens=tokens, text=text)]
    return NLPDocument(sentences=sentences, text=text)


class DummyNLP:
    def __call__(self, text: str) -> NLPDocument:
        return _doc_for_text(text)


def _build_pipeline(*_args, **_kwargs):
    return DummyNLP(), "dummy"


def _backend_factory(config: NLPConfig) -> BuiltNLPBackend:
    return BuiltNLPBackend(
        backend=DummyNLP(),
        info=NLPBackendInfo(name="fake", language=config.language),
    )


def _write_config(project_root: Path, *, group_files: str = "input/*.txt", extra: str = "") -> Path:
    (project_root / "config").mkdir(exist_ok=True)
    config_path = project_root / "config" / "groups.config.yml"
    config_path.write_text(
        "\n".join(
            [
                "groups:",
                "  text:",
                "    files:",
                f"      - {group_files}",
                "out_dir: output",
                "language: la",
                "stanza_package: perseus",
                "cpu_only: true",
                extra,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_compute_basic_features_values() -> None:
    records = [
        TokenRecord("Rosa", "rosa", "NOUN", 0),
        TokenRecord("amat", "amo", "VERB", 0),
        TokenRecord(".", ".", "PUNCT", 0),
        TokenRecord("Rosa", "rosa", "NOUN", 1),
    ]

    row = compute_basic_features(records, "Rosa amat. Rosa", "g", 1)

    assert row["sentence_count"] == 2
    assert row["token_count"] == 4
    assert row["word_token_count"] == 3
    assert row["lemma_type_count"] == 2
    assert row["mean_sentence_length"] == 1.5


def test_compute_upos_features_values() -> None:
    records = [
        TokenRecord("Rosa", "rosa", "NOUN", 0),
        TokenRecord("amat", "amo", "VERB", 0),
        TokenRecord("et", "et", "CCONJ", 0),
        TokenRecord(".", ".", "PUNCT", 0),
    ]

    row = compute_upos_features(records)

    assert row["upos_NOUN_count"] == 1
    assert row["upos_NOUN_ratio"] == pytest.approx(1 / 3)
    assert row["content_word_count"] == 2
    assert row["content_word_ratio"] == pytest.approx(2 / 3)
    assert row["function_word_count"] == 1
    assert row["function_word_ratio"] == pytest.approx(1 / 3)


def test_build_feature_rows_mfw_lemma_and_token() -> None:
    groups_texts = [
        ("g1", [Path("a.txt")], "Rosa amat et puella."),
        ("g2", [Path("b.txt")], "Rosa in villa currit."),
    ]

    lemma_rows = build_feature_rows(groups_texts, DummyNLP(), FeatureOptions(mfw=2, field="lemma"))
    token_rows = build_feature_rows(groups_texts, DummyNLP(), FeatureOptions(mfw=2, field="token"))

    assert "mfw_rosa" in lemma_rows[0]
    assert "mfw_amo" in lemma_rows[0] or "mfw_curro" in lemma_rows[0]
    assert "mfw_rosa" in token_rows[0]
    assert "mfw_amat" in token_rows[0] or "mfw_currit" in token_rows[0]


def test_write_feature_matrix_csv_and_tsv() -> None:
    rows = [{"group": "g", "token_count": 2, "mean_token_length": 4.5}]
    csv_out = io.StringIO()
    tsv_out = io.StringIO()

    write_feature_matrix(rows, csv_out, "csv")
    write_feature_matrix(rows, tsv_out, "tsv")

    assert "group,token_count,mean_token_length" in csv_out.getvalue()
    assert "group\ttoken_count\tmean_token_length" in tsv_out.getvalue()


def test_run_features_one_group_writes_csv(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("Rosa amat et puella.", encoding="utf-8")
    config_path = _write_config(tmp_path)
    out = tmp_path / "output" / "features.csv"

    rc = run_features(
        project_root=tmp_path,
        config_path=config_path,
        out=out,
        build_pipeline_fn=_build_pipeline,
        clean_mod=object(),
    )

    assert rc == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["group"] == "text"
    assert rows[0]["word_token_count"] == "4"


def test_run_features_accepts_backend_factory(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("Rosa amat et puella.", encoding="utf-8")
    config_path = _write_config(tmp_path)
    out = tmp_path / "output" / "features.csv"

    rc = run_features(
        project_root=tmp_path,
        config_path=config_path,
        out=out,
        backend_factory=_backend_factory,
        clean_mod=object(),
    )

    assert rc == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0]["group"] == "text"
    assert rows[0]["word_token_count"] == "4"


def test_run_features_two_groups_two_rows(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("Rosa amat.", encoding="utf-8")
    (tmp_path / "input" / "b.txt").write_text("puella currit.", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "groups:",
                "  a:",
                "    files: [input/a.txt]",
                "  b:",
                "    files: [input/b.txt]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "features.csv"

    run_features(
        project_root=tmp_path,
        config_path=config_path,
        out=out,
        build_pipeline_fn=_build_pipeline,
        clean_mod=object(),
    )

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [row["group"] for row in rows] == ["a", "b"]


def test_run_features_group_by_file(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("Rosa amat.", encoding="utf-8")
    (tmp_path / "input" / "b.txt").write_text("puella currit.", encoding="utf-8")
    config_path = _write_config(tmp_path)
    out = tmp_path / "features.csv"

    run_features(
        project_root=tmp_path,
        config_path=config_path,
        out=out,
        group_by_file=True,
        build_pipeline_fn=_build_pipeline,
        clean_mod=object(),
    )

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [row["group"] for row in rows] == ["a", "b"]
    assert all(row["file_count"] == "1" for row in rows)


def test_run_features_auto_single_cleaned(tmp_path: Path) -> None:
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    (cleaned / "only.cleaned.txt").write_text("Rosa amat.", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        group_files='"{cleaned_dir}/*.txt"',
        extra="\n".join(
            [
                "preprocess:",
                "  kind: cleaner",
                "  config: config/cleaner.yml",
                "grouping:",
                "  mode: auto_single_cleaned",
                "  auto_group_name: text",
            ]
        ),
    )
    (tmp_path / "config" / "cleaner.yml").write_text(
        "kind: scholastic_text\ninput: ../input\noutput: ../cleaned\n",
        encoding="utf-8",
    )
    out = tmp_path / "features.csv"

    run_features(
        project_root=tmp_path,
        config_path=config_path,
        out=out,
        build_pipeline_fn=_build_pipeline,
        clean_mod=type("Clean", (), {"main": staticmethod(lambda _argv: 0)}),
    )

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0]["group"] == "text"
    assert rows[0]["file_count"] == "1"


def test_run_features_auto_single_cleaned_errors_on_multiple(tmp_path: Path) -> None:
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    (cleaned / "a.txt").write_text("a", encoding="utf-8")
    (cleaned / "b.txt").write_text("b", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        group_files='"{cleaned_dir}/*.txt"',
        extra="\n".join(
            [
                "preprocess:",
                "  kind: cleaner",
                "  config: config/cleaner.yml",
                "grouping:",
                "  mode: auto_single_cleaned",
            ]
        ),
    )
    (tmp_path / "config" / "cleaner.yml").write_text(
        "kind: scholastic_text\ninput: ../input\noutput: ../cleaned\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected exactly one"):
        run_features(
            project_root=tmp_path,
            config_path=config_path,
            build_pipeline_fn=_build_pipeline,
            clean_mod=type("Clean", (), {"main": staticmethod(lambda _argv: 0)}),
        )


def test_exclude_lemmas_is_not_applied(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("Rosa amat.", encoding="utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "exclude_lemmas.txt").write_text("rosa\n", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        extra="\nfilters:\n  exclude_lemmas: config/exclude_lemmas.txt",
    )
    out = tmp_path / "features.csv"

    run_features(
        project_root=tmp_path,
        config_path=config_path,
        out=out,
        mfw=2,
        build_pipeline_fn=_build_pipeline,
        clean_mod=object(),
    )

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert "mfw_rosa" in rows[0]


def test_mfw_negative_errors(tmp_path: Path) -> None:
    with pytest.raises(FeatureError, match="non-negative"):
        build_feature_rows([], DummyNLP(), FeatureOptions(mfw=-1))


def test_safe_feature_name_replaces_punctuation() -> None:
    assert safe_feature_name("in-que!") == "in_que"


def test_cli_features_help() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["features", "--help"])
    assert exc.value.code == 0
