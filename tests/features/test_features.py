from __future__ import annotations

import csv
from collections import Counter
import io
from pathlib import Path

import pytest

from nlpo_toolkit.cleaner_contracts import CleanerExecutionResult
from nlpo_toolkit.corpus_analysis import cli
from nlpo_toolkit.corpus_analysis.features.errors import FeatureError
from nlpo_toolkit.corpus_analysis.features.models import (
    AnalyzedFeatureCorpus,
    CharacterNgramMode,
    CharacterNgramOptions,
    FeatureCommandResult,
    FeatureFilterPolicy,
    FeatureOptions,
    FeatureRequest,
    FeatureRow,
    FeatureSamplingOptions,
    FunctionWordOptions,
    FunctionWordSource,
    FunctionWordVocabulary,
    UposNgramOptions,
    MorphologyOptions,
)
from nlpo_toolkit.corpus_analysis.features.engine import build_feature_matrix
from nlpo_toolkit.corpus_analysis.features.filtering import (
    filter_feature_records,
    safe_feature_name,
)
from nlpo_toolkit.corpus_analysis.features.lexical import compute_basic_features
from nlpo_toolkit.corpus_analysis.features.mfw import (
    compute_mfw_features,
    select_mfw_terms,
)
from nlpo_toolkit.corpus_analysis.features.upos import compute_upos_features
from nlpo_toolkit.corpus_analysis.features.service import execute_feature_command
from nlpo_toolkit.corpus_analysis.features.function_word_loader import (
    load_function_word_vocabulary,
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
from nlpo_toolkit.corpus_analysis.corpus import PreparedCorpus
from nlpo_toolkit.corpus_analysis.ports import (
    CorpusExecutionDependencies,
    CorpusPlanningDependencies,
    CorpusPreparationDependencies,
    FeatureCommandDependencies,
    NLPExecutionDependencies,
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


def _prepared(group: str, path: Path, text: str) -> PreparedCorpus:
    return PreparedCorpus(group, (path,), text, text, Counter())


def _analyzed(
    lexical_records,
    text: str,
    *,
    raw_records=None,
    files: int = 1,
):
    source = PreparedCorpus(
        "g",
        tuple(Path(f"{index}.txt") for index in range(files)),
        text,
        text,
        Counter(),
    )
    lexical = tuple(lexical_records)
    return AnalyzedFeatureCorpus(
        source=source,
        raw_records=tuple(raw_records) if raw_records is not None else lexical,
        lexical_records=lexical,
    )


def _dependencies(cleaner=None) -> FeatureCommandDependencies:
    from nlpo_toolkit.corpus_analysis.config import load_config

    def execute_cleaner(request):
        if cleaner is None:
            raise AssertionError("cleaner service must not be called")
        return cleaner(request)

    return FeatureCommandDependencies(
        corpus=CorpusExecutionDependencies(
            planning=CorpusPlanningDependencies(
                load_config=load_config,
                cleaner_inspector=inspect_cleaner_config,
            ),
            preparation=CorpusPreparationDependencies(execute_cleaner=execute_cleaner),
        ),
        nlp=NLPExecutionDependencies(
            backend_factory=_backend_factory,
            extraction_policy=AnalysisExtractionPolicy(),
        ),
        load_function_words=load_function_word_vocabulary,
    )


def _successful_cleaner(request) -> CleanerExecutionResult:
    config = request.inspection.config
    return CleanerExecutionResult(
        config.source_path, config.kind, config.output_path, (), config.ref_tsv_path
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


def _write_config(
    project_root: Path, *, group_files: str = "input/*.txt", extra: str = ""
) -> Path:
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
        _analyzed(feature_records, "Rosa amat. Rosa", raw_records=records)
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

    analyzed = _analyzed(records, "Rosa")
    row = compute_basic_features(analyzed)

    assert row["lemma_type_count"] == 1
    assert select_mfw_terms([analyzed], count=1, field="lemma") == ("rosa",)
    assert (
        compute_mfw_features(records, terms=["rosa"], field="lemma")["mfw_rosa"] == 1.0
    )


def test_sentence_count_uses_chunk_and_sentence_index() -> None:
    records = [
        _record("a", "a", "NOUN", 0, chunk_index=0),
        _record("b", "b", "NOUN", 0, chunk_index=1),
    ]

    row = compute_basic_features(_analyzed(records, "a b"))

    assert row["sentence_count"] == 2


def test_missing_upos_and_punctuation_preserve_feature_denominators() -> None:
    records = [
        _record("Rosa", "rosa", None, 0),
        _record(".", ".", "PUNCT", 0),
    ]

    feature_records = filter_feature_records(records, policy=FeatureFilterPolicy())
    analyzed = _analyzed(feature_records, "Rosa.", raw_records=records)
    basic = compute_basic_features(analyzed)
    upos = compute_upos_features(feature_records)

    assert basic["token_count"] == 2
    assert basic["word_token_count"] == 1
    assert upos["upos_NOUN_count"] == 0
    assert upos["content_word_ratio"] == 0.0
    assert select_mfw_terms([analyzed], count=5, field="token") == ("rosa",)


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
    analyzed = _analyzed(feature_records, "xiv a rosa", raw_records=records)
    basic = compute_basic_features(analyzed)
    upos = compute_upos_features(feature_records)
    terms = select_mfw_terms([analyzed], count=3, field="token")
    mfw = compute_mfw_features(feature_records, terms=terms, field="token")

    assert [record.token for record in feature_records] == ["rosa"]
    assert basic["word_token_count"] == 1
    assert basic["lemma_type_count"] == 1
    assert upos["upos_NOUN_count"] == 1
    assert upos["upos_NUM_count"] == 0
    assert upos["upos_NOUN_ratio"] == 1.0
    assert terms == ("rosa",)
    assert mfw["mfw_rosa"] == 1.0


def test_features_roman_policy_uses_shared_surface_exceptions() -> None:
    records = (_record("vi", "vi", "NUM", 0),)

    assert (
        filter_feature_records(
            records,
            policy=FeatureFilterPolicy(drop_roman_numerals=True),
        )
        == records
    )


def test_build_feature_matrix_chunks_through_shared_extractor() -> None:
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

    rows = build_feature_matrix(
        corpora=(_prepared("g", Path("a.txt"), "Rosa amat"),),
        nlp=ChunkNLP(),
        extraction_policy=AnalysisExtractionPolicy(chunk_chars=5),
        options=FeatureOptions(),
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
    build_feature_matrix(
        corpora=(_prepared("g", Path("a.txt"), "Rosa amat"),),
        nlp=feature_backend,
        extraction_policy=policy,
        options=FeatureOptions(),
    )
    assert count_backend.calls == feature_backend.calls


def test_build_feature_matrix_mfw_lemma_and_token() -> None:
    corpora = (
        _prepared("g1", Path("a.txt"), "Rosa amat et puella."),
        _prepared("g2", Path("b.txt"), "Rosa in villa currit."),
    )

    lemma_rows = build_feature_matrix(
        corpora=corpora,
        nlp=DummyNLP(),
        extraction_policy=AnalysisExtractionPolicy(),
        options=FeatureOptions(mfw=2, field="lemma"),
    )
    token_rows = build_feature_matrix(
        corpora=corpora,
        nlp=DummyNLP(),
        extraction_policy=AnalysisExtractionPolicy(),
        options=FeatureOptions(mfw=2, field="token"),
    )

    assert "mfw_rosa" in lemma_rows[0]
    assert "mfw_amo" in lemma_rows[0] or "mfw_curro" in lemma_rows[0]
    assert "mfw_rosa" in token_rows[0]
    assert "mfw_amat" in token_rows[0] or "mfw_currit" in token_rows[0]


def test_write_feature_matrix_csv_and_tsv() -> None:
    rows = (
        FeatureRow.from_mapping(
            {"group": "g", "token_count": 2, "mean_token_length": 4.5}
        ),
    )
    csv_out = io.StringIO()
    tsv_out = io.StringIO()

    result = FeatureCommandResult(rows=rows)
    write_feature_result(result, stream=csv_out, output_format="csv")
    write_feature_result(result, stream=tsv_out, output_format="tsv")

    assert "group,token_count,mean_token_length" in csv_out.getvalue()
    assert "group\ttoken_count\tmean_token_length" in tsv_out.getvalue()


def test_feature_row_copies_and_freezes_its_mapping() -> None:
    source = {"group": "g", "token_count": 2}
    row = FeatureRow.from_mapping(source)
    source["token_count"] = 3
    assert row["token_count"] == 2
    with pytest.raises(TypeError):
        row.values["token_count"] = 4  # type: ignore[index]


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
            corpus=base.corpus,
            nlp=NLPExecutionDependencies(
                backend_factory=recording_factory,
                extraction_policy=base.nlp.extraction_policy,
            ),
            load_function_words=base.load_function_words,
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


def test_configured_per_file_grouping_supports_fixed_windows(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("Rosa amat", encoding="utf-8")
    (tmp_path / "input" / "b.txt").write_text("puella currit", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        extra="grouping:\n  mode: per_file",
    )
    result = execute_feature_command(
        FeatureRequest(
            corpus=CorpusPreparationRequest(tmp_path, config_path),
            sampling=FeatureSamplingOptions(window_tokens=2),
        ),
        dependencies=_dependencies(),
    )
    assert [row["group"] for row in result.rows] == ["a", "b"]
    assert all(row["word_token_count"] == 2 for row in result.rows)


def test_feature_command_applies_one_shared_filter_and_loads_roman_exceptions_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import nlpo_toolkit.corpus_analysis.features.engine as features_mod
    import nlpo_toolkit.corpus_analysis.execution_session as session_mod

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
    real_loader = session_mod.load_roman_exceptions
    real_filter = features_mod.filter_feature_records

    def recording_loader(path: Path) -> frozenset[str]:
        load_calls.append(path)
        return real_loader(path)

    def recording_filter(records, *, policy):
        records = tuple(records)
        filter_calls.append(records)
        return real_filter(records, policy=policy)

    monkeypatch.setattr(session_mod, "load_roman_exceptions", recording_loader)
    monkeypatch.setattr(features_mod, "filter_feature_records", recording_filter)

    dependencies = _dependencies()
    dependencies = FeatureCommandDependencies(
        corpus=dependencies.corpus,
        nlp=NLPExecutionDependencies(
            backend_factory=lambda config: BuiltNLPBackend(
                backend=nlp,
                info=NLPBackendInfo(name="fake", language=config.language),
            ),
            extraction_policy=dependencies.nlp.extraction_policy,
        ),
        load_function_words=dependencies.load_function_words,
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
        dependencies=_dependencies(_successful_cleaner),
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
            dependencies=_dependencies(_successful_cleaner),
        )


def test_mfw_negative_errors(tmp_path: Path) -> None:
    with pytest.raises(FeatureError, match="non-negative"):
        build_feature_matrix(
            corpora=(),
            nlp=DummyNLP(),
            extraction_policy=AnalysisExtractionPolicy(),
            options=FeatureOptions(mfw=-1),
        )


def test_safe_feature_name_replaces_punctuation() -> None:
    assert safe_feature_name("in-que!") == "in_que"


def test_cli_features_help(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["features", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--window-tokens" in help_text
    assert "--step-tokens" in help_text
    assert "--include-partial-window" in help_text
    assert "--lexical-diversity" in help_text
    assert "--lexdiv-window" in help_text
    assert "--mtld-threshold" in help_text
    assert "--hdd-sample-size" in help_text
    assert "--function-words" in help_text
    assert "--function-word-field" in help_text
    assert "--morphology" in help_text
    assert "--morph-attribute" in help_text
    assert "--morph-bundle-top" in help_text


def test_cli_sampling_arguments_are_composed_into_request(monkeypatch) -> None:
    import nlpo_toolkit.corpus_analysis.cli.features as feature_cli

    requests: list[FeatureRequest] = []

    def execute(request, *, dependencies):
        requests.append(request)
        return FeatureCommandResult(())

    monkeypatch.setattr(feature_cli, "execute_feature_command", execute)
    monkeypatch.setattr(
        feature_cli, "default_feature_command_dependencies", lambda: object()
    )
    stdout = io.StringIO()
    stderr = io.StringIO()
    assert (
        cli.main(
            [
                "features",
                "--window-tokens",
                "1000",
                "--step-tokens",
                "500",
                "--include-partial-window",
            ],
            stdout=stdout,
            stderr=stderr,
        )
        == 0
    )
    assert requests[0].sampling == FeatureSamplingOptions(1000, 500, True)


def test_cli_invalid_sampling_value_returns_one() -> None:
    stderr = io.StringIO()
    assert (
        cli.main(
            ["features", "--window-tokens", "0"],
            stdout=io.StringIO(),
            stderr=stderr,
        )
        == 1
    )
    assert "window_tokens must be a positive integer" in stderr.getvalue()


def test_cli_function_word_arguments_are_composed_separately_from_mfw(
    monkeypatch,
) -> None:
    import nlpo_toolkit.corpus_analysis.cli.features as feature_cli

    requests: list[FeatureRequest] = []

    def execute(request, *, dependencies):
        requests.append(request)
        return FeatureCommandResult(())

    monkeypatch.setattr(feature_cli, "execute_feature_command", execute)
    monkeypatch.setattr(
        feature_cli, "default_feature_command_dependencies", lambda: object()
    )
    assert (
        cli.main(
            [
                "features",
                "--mfw",
                "5",
                "--field",
                "token",
                "--function-words",
                "config/function_words.txt",
                "--function-word-field",
                "lemma",
            ],
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        == 0
    )
    assert requests[0].field == "token"
    assert requests[0].function_words == FunctionWordSource(
        Path("config/function_words.txt"), "lemma"
    )

    requests.clear()
    assert (
        cli.main(
            ["features", "--function-words", "config/function_words.txt"],
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        == 0
    )
    assert requests[0].function_words == FunctionWordSource(
        Path("config/function_words.txt"), "lemma"
    )


def test_cli_function_word_field_requires_file() -> None:
    stderr = io.StringIO()
    assert (
        cli.main(
            ["features", "--function-word-field", "token"],
            stdout=io.StringIO(),
            stderr=stderr,
        )
        == 1
    )
    assert "--function-word-field requires --function-words" in stderr.getvalue()


def test_cli_character_ngram_arguments_build_one_options_model(monkeypatch) -> None:
    import nlpo_toolkit.corpus_analysis.cli.features as feature_cli

    requests: list[FeatureRequest] = []

    def execute(request, *, dependencies):
        requests.append(request)
        return FeatureCommandResult(())

    monkeypatch.setattr(feature_cli, "execute_feature_command", execute)
    monkeypatch.setattr(
        feature_cli, "default_feature_command_dependencies", lambda: object()
    )
    assert (
        cli.main(
            [
                "features",
                "--char-ngram-size",
                "3",
                "--char-ngram-size",
                "5",
                "--char-ngram-mode",
                "letters-only",
                "--char-ngram-mode",
                "full",
            ],
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        == 0
    )
    assert requests[0].character_ngrams == CharacterNgramOptions(
        (3, 5),
        500,
        (CharacterNgramMode.LETTERS_ONLY, CharacterNgramMode.FULL),
    )


def test_cli_character_ngram_top_requires_size_and_duplicates_fail() -> None:
    stderr = io.StringIO()
    assert (
        cli.main(
            ["features", "--char-ngram-top", "10"],
            stdout=io.StringIO(),
            stderr=stderr,
        )
        == 1
    )
    assert "requires --char-ngram-size" in stderr.getvalue()
    stderr = io.StringIO()
    assert (
        cli.main(
            ["features", "--char-ngram-size", "3", "--char-ngram-size", "3"],
            stdout=io.StringIO(),
            stderr=stderr,
        )
        == 1
    )
    assert "duplicate character n-gram size: 3" in stderr.getvalue()


def test_cli_character_ngram_mode_requires_size_and_rejects_duplicates() -> None:
    stderr = io.StringIO()
    assert (
        cli.main(
            ["features", "--char-ngram-mode", "letters-only"],
            stdout=io.StringIO(),
            stderr=stderr,
        )
        == 1
    )
    assert "--char-ngram-mode requires --char-ngram-size" in stderr.getvalue()
    stderr = io.StringIO()
    assert (
        cli.main(
            [
                "features",
                "--char-ngram-size",
                "3",
                "--char-ngram-mode",
                "full",
                "--char-ngram-mode",
                "full",
            ],
            stdout=io.StringIO(),
            stderr=stderr,
        )
        == 1
    )
    assert "duplicate --char-ngram-mode: full" in stderr.getvalue()


def test_cli_upos_ngram_arguments_build_one_options_model(monkeypatch) -> None:
    import nlpo_toolkit.corpus_analysis.cli.features as feature_cli

    requests: list[FeatureRequest] = []

    def execute(request, *, dependencies):
        requests.append(request)
        return FeatureCommandResult(())

    monkeypatch.setattr(feature_cli, "execute_feature_command", execute)
    monkeypatch.setattr(
        feature_cli, "default_feature_command_dependencies", lambda: object()
    )
    assert (
        cli.main(
            [
                "features",
                "--upos-ngram-size",
                "3",
                "--upos-ngram-size",
                "2",
            ],
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        == 0
    )
    assert requests[0].upos_ngrams == UposNgramOptions((3, 2), 100)


def test_cli_upos_ngram_top_requires_size_and_duplicates_fail() -> None:
    stderr = io.StringIO()
    assert (
        cli.main(
            ["features", "--upos-ngram-top", "10"],
            stdout=io.StringIO(),
            stderr=stderr,
        )
        == 1
    )
    assert "requires --upos-ngram-size" in stderr.getvalue()
    stderr = io.StringIO()
    assert (
        cli.main(
            ["features", "--upos-ngram-size", "2", "--upos-ngram-size", "2"],
            stdout=io.StringIO(),
            stderr=stderr,
        )
        == 1
    )
    assert "duplicate UPOS n-gram size: 2" in stderr.getvalue()
    stderr = io.StringIO()
    assert (
        cli.main(
            ["features", "--upos-ngram-size", "4"],
            stdout=io.StringIO(),
            stderr=stderr,
        )
        == 1
    )
    assert "--upos-ngram-size must be 2 or 3" in stderr.getvalue()


def test_cli_morphology_arguments_build_typed_options(monkeypatch) -> None:
    import nlpo_toolkit.corpus_analysis.cli.features as feature_cli

    requests: list[FeatureRequest] = []

    def execute(request, *, dependencies):
        requests.append(request)
        return FeatureCommandResult(())

    monkeypatch.setattr(feature_cli, "execute_feature_command", execute)
    monkeypatch.setattr(
        feature_cli, "default_feature_command_dependencies", lambda: object()
    )
    assert cli.main(
        [
            "features", "--morph-attribute", "Case", "--morph-attribute",
            "Number", "--morph-bundle-top", "10",
        ],
        stdout=io.StringIO(), stderr=io.StringIO(),
    ) == 0
    assert requests[0].morphology == MorphologyOptions(
        True, ("Case", "Number"), 10
    )


def test_cli_morphology_rejects_duplicate_attribute_and_invalid_top() -> None:
    for arguments, message in (
        (("--morph-attribute", "Case", "--morph-attribute", "Case"), "duplicate"),
        (("--morph-bundle-top", "0"), "positive integer"),
    ):
        stderr = io.StringIO()
        assert cli.main(
            ["features", *arguments], stdout=io.StringIO(), stderr=stderr
        ) == 1
        assert message in stderr.getvalue()


def test_cli_function_word_validation_failure_does_not_create_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "features.csv"
    stderr = io.StringIO()

    assert (
        cli.main(
            [
                "features",
                "--project-root",
                str(tmp_path),
                "--function-words",
                "missing.txt",
                "--out",
                str(output),
            ],
            stdout=io.StringIO(),
            stderr=stderr,
        )
        == 1
    )
    assert "function-word file not found" in stderr.getvalue()
    assert not output.exists()


def test_function_word_loader_runs_before_backend_and_resolves_project_path(
    tmp_path: Path,
) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("Rosa amat et", encoding="utf-8")
    config_path = _write_config(tmp_path)
    loaded_paths: list[Path] = []
    backend_calls = 0

    def loader(path: Path) -> FunctionWordVocabulary:
        loaded_paths.append(path)
        return FunctionWordVocabulary(("et",))

    def backend_factory(config: NLPConfig) -> BuiltNLPBackend:
        nonlocal backend_calls
        backend_calls += 1
        return _backend_factory(config)

    base = _dependencies()
    result = execute_feature_command(
        FeatureRequest(
            corpus=CorpusPreparationRequest(tmp_path, config_path),
            function_words=FunctionWordSource(Path("lists/function_words.txt")),
        ),
        dependencies=FeatureCommandDependencies(
            corpus=base.corpus,
            nlp=NLPExecutionDependencies(
                backend_factory=backend_factory,
                extraction_policy=base.nlp.extraction_policy,
            ),
            load_function_words=loader,
        ),
    )

    assert loaded_paths == [(tmp_path / "lists/function_words.txt").resolve()]
    assert backend_calls == 1
    assert result.rows[0]["fw_et"] == pytest.approx(1 / 3)


def test_function_word_validation_failure_is_before_backend_start(
    tmp_path: Path,
) -> None:
    backend_calls = 0

    def failing_loader(path: Path) -> FunctionWordVocabulary:
        raise FeatureError(f"invalid function-word list: {path}")

    def backend_factory(config: NLPConfig) -> BuiltNLPBackend:
        nonlocal backend_calls
        backend_calls += 1
        return _backend_factory(config)

    base = _dependencies()
    with pytest.raises(FeatureError, match="invalid function-word list"):
        execute_feature_command(
            FeatureRequest(
                corpus=CorpusPreparationRequest(tmp_path, tmp_path / "missing.yml"),
                function_words=FunctionWordSource(Path("bad.txt")),
            ),
            dependencies=FeatureCommandDependencies(
                corpus=base.corpus,
                nlp=NLPExecutionDependencies(
                    backend_factory=backend_factory,
                    extraction_policy=base.nlp.extraction_policy,
                ),
                load_function_words=failing_loader,
            ),
        )
    assert backend_calls == 0


def test_function_words_follow_upos_and_precede_mfw_in_each_sample() -> None:
    rows = build_feature_matrix(
        corpora=(_prepared("g", Path("a.txt"), "et rosa et"),),
        nlp=DummyNLP(),
        extraction_policy=AnalysisExtractionPolicy(),
        options=FeatureOptions(
            mfw=1,
            field="token",
            sampling=FeatureSamplingOptions(2, 1, include_partial=True),
            function_words=FunctionWordOptions(
                FunctionWordVocabulary(("et", "non")),
                field="token",
            ),
        ),
    )

    first = rows[0]
    keys = tuple(first)
    assert keys.index("function_word_ratio") < keys.index("fw_et")
    assert keys.index("fw_non") < keys.index("mfw_et")
    assert first["fw_et"] == 0.5
    assert first["fw_non"] == 0.0
    assert rows[1]["fw_et"] == 0.5
    assert rows[-1]["fw_et"] == 1.0


def test_character_ngrams_use_prepared_text_and_precede_mfw() -> None:
    rows = build_feature_matrix(
        corpora=(_prepared("g", Path("a.txt"), "Rosa, amat."),),
        nlp=DummyNLP(),
        extraction_policy=AnalysisExtractionPolicy(),
        options=FeatureOptions(
            mfw=1,
            field="token",
            character_ngrams=CharacterNgramOptions((3,), top=4),
        ),
    )
    keys = tuple(rows[0])
    character_columns = tuple(key for key in keys if key.startswith("char3_"))
    assert len(character_columns) == 4
    assert keys.index(character_columns[-1]) < keys.index("mfw_amat")
    assert any("_u00002c_" in key for key in character_columns)


def test_upos_ngrams_follow_unigrams_and_precede_other_vocabularies() -> None:
    rows = build_feature_matrix(
        corpora=(_prepared("g", Path("a.txt"), "rosa amat et puella"),),
        nlp=DummyNLP(),
        extraction_policy=AnalysisExtractionPolicy(),
        options=FeatureOptions(
            mfw=1,
            field="token",
            upos_ngrams=UposNgramOptions((2, 3), top=2),
            function_words=FunctionWordOptions(
                FunctionWordVocabulary(("et",)), field="token"
            ),
        ),
    )
    keys = tuple(rows[0])
    upos2 = tuple(key for key in keys if key.startswith("upos2_"))
    upos3 = tuple(key for key in keys if key.startswith("upos3_"))
    assert len(upos2) == len(upos3) == 2
    assert keys.index("function_word_ratio") < keys.index(upos2[0])
    assert keys.index(upos3[-1]) < keys.index("fw_et")
    assert keys.index("fw_et") < keys.index("mfw_amat")


def test_upos_vocabulary_is_selected_before_overlapping_samples() -> None:
    corpus = (_prepared("g", Path("a.txt"), "rosa amat et puella"),)
    first = build_feature_matrix(
        corpora=corpus,
        nlp=DummyNLP(),
        extraction_policy=AnalysisExtractionPolicy(),
        options=FeatureOptions(
            sampling=FeatureSamplingOptions(3, 1, include_partial=True),
            upos_ngrams=UposNgramOptions((2,), top=3),
        ),
    )
    second = build_feature_matrix(
        corpora=corpus,
        nlp=DummyNLP(),
        extraction_policy=AnalysisExtractionPolicy(),
        options=FeatureOptions(
            sampling=FeatureSamplingOptions(2, 2, include_partial=True),
            upos_ngrams=UposNgramOptions((2,), top=3),
        ),
    )

    def columns(row: FeatureRow) -> tuple[str, ...]:
        return tuple(key for key in row if key.startswith("upos2_"))

    assert columns(first[0]) == columns(second[0])


def test_character_file_boundary_failure_precedes_backend_start(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("Rosa amat", encoding="utf-8")
    (tmp_path / "input" / "b.txt").write_text("Puella currit", encoding="utf-8")
    config_path = _write_config(tmp_path)
    backend_calls = 0

    def backend_factory(config: NLPConfig) -> BuiltNLPBackend:
        nonlocal backend_calls
        backend_calls += 1
        return _backend_factory(config)

    base = _dependencies()
    with pytest.raises(FeatureError, match="one source file"):
        execute_feature_command(
            FeatureRequest(
                corpus=CorpusPreparationRequest(tmp_path, config_path),
                character_ngrams=CharacterNgramOptions((3,)),
            ),
            dependencies=FeatureCommandDependencies(
                corpus=base.corpus,
                nlp=NLPExecutionDependencies(
                    backend_factory=backend_factory,
                    extraction_policy=base.nlp.extraction_policy,
                ),
                load_function_words=base.load_function_words,
            ),
        )
    assert backend_calls == 0
