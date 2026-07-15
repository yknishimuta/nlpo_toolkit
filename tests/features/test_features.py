from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis import cli
from nlpo_toolkit.corpus_analysis.features import (
    FeatureError,
    FeatureCommandResult,
    FeatureFilterPolicy,
    FeatureOptions,
    FeatureRequest,
    build_feature_rows,
    compute_basic_features,
    compute_mfw_features,
    compute_upos_features,
    select_mfw,
    execute_feature_command,
    filter_feature_records,
    safe_feature_name,
)
from nlpo_toolkit.corpus_analysis.requests import CorpusPreparationRequest
from nlpo_toolkit.corpus_analysis.cli.output import write_feature_result
from nlpo_toolkit.corpus_analysis.analysis_records import (
    NLPAnalysisRecord,
    iter_nlp_analysis_records_from_text,
)
from nlpo_toolkit.corpus_analysis.analysis_policy import AnalysisExtractionPolicy
from nlpo_toolkit.nlp.contracts import (
    BuiltNLPBackend,
    NLPBackendInfo,
    NLPDocument,
    NLPSentence,
    NLPToken,
)
from nlpo_toolkit.corpus_analysis.config import NLPConfig
from nlpo_toolkit.corpus_analysis.ports import (
    AnalysisDependencies,
    CorpusPlanningDependencies,
    FeatureCommandDependencies,
)
from nlpo_toolkit.latin.cleaners.config_loader import inspect_cleaner_config


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


def _backend_factory(config: NLPConfig) -> BuiltNLPBackend:
    return BuiltNLPBackend(
        backend=DummyNLP(),
        info=NLPBackendInfo(name="fake", language=config.language),
    )


def _dependencies(cleaner=None) -> FeatureCommandDependencies:
    from nlpo_toolkit.corpus_analysis.config import load_config

    def cleaner_loader():
        if cleaner is None:
            raise AssertionError("cleaner loader must not be called")
        return cleaner

    return FeatureCommandDependencies(
        planning=CorpusPlanningDependencies(
            load_config=load_config,
            cleaner_loader=cleaner_loader,
            cleaner_inspector=inspect_cleaner_config,
        ),
        analysis=AnalysisDependencies(
            backend_factory=_backend_factory,
            extraction_policy=AnalysisExtractionPolicy(),
        ),
    )


def _record(
    token: str,
    lemma: str | None,
    upos: str | None,
    sentence_index: int,
    *,
    chunk_index: int = 0,
    token_index: int = 0,
    global_token_index: int = 0,
) -> NLPAnalysisRecord:
    return NLPAnalysisRecord(
        chunk_index=chunk_index,
        sentence_index=sentence_index,
        token_index=token_index,
        global_token_index=global_token_index,
        char_start_in_chunk=None,
        char_end_in_chunk=None,
        char_start_in_text=None,
        char_end_in_text=None,
        sentence="",
        token=token,
        lemma=lemma,
        upos=upos,
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
                "nlp:",
                "  language: la",
                "  stanza_package: perseus",
                "  cpu_only: true",
                extra,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_compute_basic_features_values() -> None:
    records = [
        _record("Rosa", "rosa", "NOUN", 0),
        _record("amat", "amo", "VERB", 0),
        _record(".", ".", "PUNCT", 0),
        _record("Rosa", "rosa", "NOUN", 1),
    ]

    feature_records = filter_feature_records(records, policy=FeatureFilterPolicy())
    row = compute_basic_features(
        feature_records,
        "Rosa amat. Rosa",
        "g",
        1,
        raw_token_count=len(records),
        sentence_count=2,
    )

    assert row["sentence_count"] == 2
    assert row["token_count"] == 4
    assert row["word_token_count"] == 3
    assert row["lemma_type_count"] == 2
    assert row["mean_sentence_length"] == 1.5


def test_compute_upos_features_values() -> None:
    records = [
        _record("Rosa", "rosa", "NOUN", 0),
        _record("amat", "amo", "VERB", 0),
        _record("et", "et", "CCONJ", 0),
        _record(".", ".", "PUNCT", 0),
    ]

    row = compute_upos_features(
        filter_feature_records(records, policy=FeatureFilterPolicy())
    )

    assert row["upos_NOUN_count"] == 1
    assert row["upos_NOUN_ratio"] == pytest.approx(1 / 3)
    assert row["content_word_count"] == 2
    assert row["content_word_ratio"] == pytest.approx(2 / 3)
    assert row["function_word_count"] == 1
    assert row["function_word_ratio"] == pytest.approx(1 / 3)


def test_missing_lemma_falls_back_to_surface_for_basic_and_mfw() -> None:
    records = [_record("Rosa", None, "NOUN", 0)]

    row = compute_basic_features(
        records,
        "Rosa",
        "g",
        1,
        raw_token_count=1,
        sentence_count=1,
    )

    assert row["lemma_type_count"] == 1
    assert select_mfw([records], 1, "lemma") == ["rosa"]
    assert compute_mfw_features(records, ["rosa"], "lemma")["mfw_rosa"] == 1.0


def test_sentence_count_uses_chunk_and_sentence_index() -> None:
    records = [
        _record("a", "a", "NOUN", 0, chunk_index=0),
        _record("b", "b", "NOUN", 0, chunk_index=1),
    ]

    row = compute_basic_features(
        records,
        "a b",
        "g",
        1,
        raw_token_count=2,
        sentence_count=2,
    )

    assert row["sentence_count"] == 2


def test_missing_upos_and_punctuation_preserve_feature_denominators() -> None:
    records = [
        _record("Rosa", "rosa", None, 0),
        _record(".", ".", "PUNCT", 0),
    ]

    feature_records = filter_feature_records(records, policy=FeatureFilterPolicy())
    basic = compute_basic_features(
        feature_records,
        "Rosa.",
        "g",
        1,
        raw_token_count=2,
        sentence_count=1,
    )
    upos = compute_upos_features(feature_records)

    assert basic["token_count"] == 2
    assert basic["word_token_count"] == 1
    assert upos["upos_NOUN_count"] == 0
    assert upos["content_word_ratio"] == 0.0
    assert select_mfw([feature_records], 5, "token") == ["rosa"]


def test_feature_filter_is_shared_by_basic_upos_and_mfw() -> None:
    records = [
        _record("xiv", "xiv", "NUM", 0),
        _record("a", "a", "NOUN", 0),
        _record("rosa", "rosa", "NOUN", 0),
        _record(".", ".", "PUNCT", 0),
    ]
    feature_records = filter_feature_records(
        records,
        policy=FeatureFilterPolicy(
            min_token_length=2,
            drop_roman_numerals=True,
        ),
    )
    basic = compute_basic_features(
        feature_records,
        "xiv a rosa",
        "g",
        1,
        raw_token_count=4,
        sentence_count=1,
    )
    upos = compute_upos_features(feature_records)
    terms = select_mfw([feature_records], 3, "token")
    mfw = compute_mfw_features(feature_records, terms, "token")

    assert [record.token for record in feature_records] == ["rosa"]
    assert basic["word_token_count"] == 1
    assert basic["lemma_type_count"] == 1
    assert upos["upos_NOUN_count"] == 1
    assert upos["upos_NUM_count"] == 0
    assert upos["upos_NOUN_ratio"] == 1.0
    assert terms == ["rosa"]
    assert mfw["mfw_rosa"] == 1.0


def test_features_roman_policy_uses_shared_surface_exceptions() -> None:
    records = (_record("vi", "vi", "NUM", 0),)

    assert filter_feature_records(
        records,
        policy=FeatureFilterPolicy(drop_roman_numerals=True),
    ) == records


def test_build_feature_rows_chunks_through_shared_extractor() -> None:
    calls: list[str] = []

    class ChunkNLP:
        def __call__(self, text: str) -> NLPDocument:
            calls.append(text)
            token = text.strip()
            return NLPDocument(
                sentences=[
                    NLPSentence(
                        tokens=[NLPToken(token, token.lower(), "NOUN")],
                        text=text,
                    )
                ],
                text=text,
            )

    rows = build_feature_rows(
        [("g", [Path("a.txt")], "Rosa amat")],
        ChunkNLP(),
        FeatureOptions(extraction_policy=AnalysisExtractionPolicy(chunk_chars=5)),
    )

    assert len(calls) == 2
    assert rows[0]["token_count"] == 2
    assert rows[0]["sentence_count"] == 2


def test_count_and_features_share_chunk_boundaries() -> None:
    policy = AnalysisExtractionPolicy(chunk_chars=5)

    class RecordingNLP:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def __call__(self, text: str) -> NLPDocument:
            self.calls.append(text)
            token = text.strip()
            return NLPDocument(
                sentences=[
                    NLPSentence(
                        tokens=[NLPToken(token, token.lower(), "NOUN")],
                        text=text,
                    )
                ],
                text=text,
            )

    count_backend = RecordingNLP()
    feature_backend = RecordingNLP()
    list(
        iter_nlp_analysis_records_from_text(
            text="Rosa amat",
            nlp=count_backend,
            policy=policy,
        )
    )
    build_feature_rows(
        [("g", [Path("a.txt")], "Rosa amat")],
        feature_backend,
        FeatureOptions(extraction_policy=policy),
    )
    assert count_backend.calls == feature_backend.calls


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

    result = FeatureCommandResult(rows=tuple(rows))
    write_feature_result(result, stream=csv_out, output_format="csv")
    write_feature_result(result, stream=tsv_out, output_format="tsv")

    assert "group,token_count,mean_token_length" in csv_out.getvalue()
    assert "group\ttoken_count\tmean_token_length" in tsv_out.getvalue()


def test_run_features_one_group_writes_csv(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("Rosa amat et puella.", encoding="utf-8")
    config_path = _write_config(tmp_path)
    out = tmp_path / "output" / "features.csv"

    result = execute_feature_command(
        FeatureRequest(
            corpus=CorpusPreparationRequest(tmp_path, config_path),
        ),
        dependencies=_dependencies(),
    )

    out.parent.mkdir()
    with out.open("w", encoding="utf-8", newline="") as stream:
        write_feature_result(result, stream=stream, output_format="csv")
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["group"] == "text"
    assert rows[0]["word_token_count"] == "4"


def test_run_features_accepts_backend_factory(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("Rosa amat et puella.", encoding="utf-8")
    config_path = _write_config(tmp_path)
    out = tmp_path / "output" / "features.csv"
    calls: list[NLPConfig] = []

    def recording_factory(config: NLPConfig) -> BuiltNLPBackend:
        calls.append(config)
        return _backend_factory(config)

    base = _dependencies()

    result = execute_feature_command(
        FeatureRequest(
            corpus=CorpusPreparationRequest(tmp_path, config_path),
        ),
        dependencies=FeatureCommandDependencies(
            planning=base.planning,
            analysis=AnalysisDependencies(
                backend_factory=recording_factory,
                extraction_policy=base.analysis.extraction_policy,
            ),
        ),
    )

    out.parent.mkdir()
    with out.open("w", encoding="utf-8", newline="") as stream:
        write_feature_result(result, stream=stream, output_format="csv")
    assert len(calls) == 1
    assert calls[0].language == "la"
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

    result = execute_feature_command(
        FeatureRequest(
            corpus=CorpusPreparationRequest(tmp_path, config_path),
        ),
        dependencies=_dependencies(),
    )

    with out.open("w", encoding="utf-8", newline="") as stream:
        write_feature_result(result, stream=stream, output_format="csv")

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [row["group"] for row in rows] == ["a", "b"]


def test_feature_command_applies_one_shared_filter_and_loads_roman_exceptions_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import nlpo_toolkit.corpus_analysis.features as features_mod

    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("xiv a rosa", encoding="utf-8")
    (tmp_path / "input" / "b.txt").write_text("xiv a rosa", encoding="utf-8")
    (tmp_path / "config").mkdir()
    exceptions_path = tmp_path / "config" / "roman.txt"
    exceptions_path.write_text("XIV\n", encoding="utf-8")
    config_path = tmp_path / "groups.yml"
    config_path.write_text(
        "groups:\n"
        "  a: {files: [input/a.txt]}\n"
        "  b: {files: [input/b.txt]}\n"
        "filters:\n"
        "  min_token_length: 2\n"
        "  drop_roman_numerals: true\n"
        "  roman_exceptions_file: config/roman.txt\n"
        "  upos_targets: [VERB]\n",
        encoding="utf-8",
    )

    class RecordingNLP:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def __call__(self, text: str) -> NLPDocument:
            self.calls.append(text)
            return NLPDocument(
                sentences=[
                    NLPSentence(
                        text=text,
                        tokens=[
                            NLPToken("xiv", "fourteen", "NUM"),
                            NLPToken("a", "a", "NOUN"),
                            NLPToken("rosa", "rosa", "NOUN"),
                        ],
                    )
                ],
                text=text,
            )

    nlp = RecordingNLP()
    load_calls: list[Path] = []
    filter_calls: list[tuple[NLPAnalysisRecord, ...]] = []
    real_loader = features_mod.load_roman_exceptions
    real_filter = features_mod.filter_feature_records

    def recording_loader(path: Path) -> frozenset[str]:
        load_calls.append(path)
        return real_loader(path)

    def recording_filter(records, *, policy):
        records = tuple(records)
        filter_calls.append(records)
        return real_filter(records, policy=policy)

    monkeypatch.setattr(features_mod, "load_roman_exceptions", recording_loader)
    monkeypatch.setattr(features_mod, "filter_feature_records", recording_filter)

    dependencies = _dependencies()
    dependencies = FeatureCommandDependencies(
        planning=dependencies.planning,
        analysis=AnalysisDependencies(
            backend_factory=lambda config: BuiltNLPBackend(
                backend=nlp,
                info=NLPBackendInfo(name="fake", language=config.language),
            ),
            extraction_policy=dependencies.analysis.extraction_policy,
        ),
    )
    result = execute_feature_command(
        FeatureRequest(
            corpus=CorpusPreparationRequest(tmp_path, config_path),
            mfw=2,
            field="lemma",
        ),
        dependencies=dependencies,
    )

    assert load_calls == [exceptions_path.resolve()]
    assert len(filter_calls) == 2
    assert len(nlp.calls) == 2
    assert [row["group"] for row in result.rows] == ["a", "b"]
    for row in result.rows:
        assert row["token_count"] == 3
        assert row["word_token_count"] == 2
        assert row["upos_NUM_count"] == 1
        assert row["upos_NOUN_count"] == 1
        assert row["mfw_fourteen"] == 0.5
        assert row["mfw_rosa"] == 0.5


def test_run_features_group_by_file(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("Rosa amat.", encoding="utf-8")
    (tmp_path / "input" / "b.txt").write_text("puella currit.", encoding="utf-8")
    config_path = _write_config(tmp_path)
    out = tmp_path / "features.csv"

    result = execute_feature_command(
        FeatureRequest(
            corpus=CorpusPreparationRequest(
                tmp_path, config_path, grouping_override="per_file"
            ),
        ),
        dependencies=_dependencies(),
    )

    with out.open("w", encoding="utf-8", newline="") as stream:
        write_feature_result(result, stream=stream, output_format="csv")

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [row["group"] for row in rows] == ["a", "b"]
    assert all(row["file_count"] == "1" for row in rows)


def test_features_does_not_apply_count_partition_validation(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("Rosa amat.", encoding="utf-8")
    config_path = tmp_path / "config.yml"
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

    result = execute_feature_command(
        FeatureRequest(
            corpus=CorpusPreparationRequest(
                tmp_path, config_path, grouping_override="per_file"
            ),
        ),
        dependencies=_dependencies(),
    )

    assert len(result.rows) == 1


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
            "kind: scholastic_text\ninput: ../cleaned\noutput: ../cleaned\n",
        encoding="utf-8",
    )
    out = tmp_path / "features.csv"

    result = execute_feature_command(
        FeatureRequest(
            corpus=CorpusPreparationRequest(tmp_path, config_path),
        ),
        dependencies=_dependencies(
            type("Clean", (), {"main": staticmethod(lambda _argv: 0)})
        ),
    )

    with out.open("w", encoding="utf-8", newline="") as stream:
        write_feature_result(result, stream=stream, output_format="csv")

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
            "kind: scholastic_text\ninput: ../cleaned\noutput: ../cleaned\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected exactly one"):
        execute_feature_command(
            FeatureRequest(
                corpus=CorpusPreparationRequest(tmp_path, config_path),
            ),
            dependencies=_dependencies(
                type("Clean", (), {"main": staticmethod(lambda _argv: 0)})
            ),
        )


def test_mfw_negative_errors(tmp_path: Path) -> None:
    with pytest.raises(FeatureError, match="non-negative"):
        build_feature_rows([], DummyNLP(), FeatureOptions(mfw=-1))


def test_safe_feature_name_replaces_punctuation() -> None:
    assert safe_feature_name("in-que!") == "in_que"


def test_cli_features_help() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["features", "--help"])
    assert exc.value.code == 0
