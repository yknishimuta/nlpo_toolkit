from __future__ import annotations

from tests.corpus_analysis.fake_nlp import FakeNLPBackend, corpus_request

from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import cast, Iterator

import pytest

from nlpo_toolkit.nlp.contracts import BuiltNLPBackend, NLPBackendInfo
from nlpo_toolkit.corpus_analysis.analysis_orchestration import analyze_corpora
from nlpo_toolkit.corpus_analysis.analysis_cache_stats import AnalysisCacheStatsCollector
from nlpo_toolkit.corpus_analysis.analysis_results import AnalysisResults, GroupAnalysisResult
from nlpo_toolkit.corpus_analysis.analysis_cache_results import AnalysisRecordCacheStatus
from nlpo_toolkit.corpus_analysis.analysis_records import NLPAnalysisRecord
from nlpo_toolkit.corpus_analysis.ports import AnalysisRecordRequest, AnalysisRecordSource
from nlpo_toolkit.corpus_analysis.postprocessing.lemma_normalization import apply_lemma_normalization
from nlpo_toolkit.corpus_analysis.postprocessing.dictionary import classify_dictionary_entries
from nlpo_toolkit.corpus_analysis.corpus import PreparedCorpus
from nlpo_toolkit.corpus_analysis.config import ensure_app_config
from nlpo_toolkit.corpus_analysis.reporting.environment import collect_runtime_environment
from nlpo_toolkit.corpus_analysis.reporting.metadata import build_run_metadata, run_metadata_to_json_value
from nlpo_toolkit.corpus_analysis.reporting.summary import render_run_summary
from nlpo_toolkit.corpus_analysis.comparison_run_results import (
    ConfiguredComparisonsRunResult,
)
from nlpo_toolkit.corpus_analysis.partition_run_results import (
    PartitionValidationRunResult,
)
from nlpo_toolkit.corpus_analysis.runtime import prepare_run_context
from nlpo_toolkit.nlp.contracts import NLPDocument, NLPSentence, NLPToken
from tests.corpus_analysis.fake_nlp import runner_dependencies
from tests.corpus_analysis.fake_publication import (
    InMemoryRecordArtifactSessionFactory,
    RecordingGroupArtifactPublisher,
    recording_publication_dependencies,
)


class FakeBackend:
    def __call__(self, text: str) -> NLPDocument:
        return NLPDocument(
            text=text,
            sentences=[
                NLPSentence(
                    text=text,
                    tokens=[
                        NLPToken("item_a", "item_a", "NOUN", 0, 6),
                        NLPToken("item_a", "item_a", "NOUN", 7, 13),
                        NLPToken("item_b", "item_b", "NOUN", 14, 20),
                    ],
                )
            ],
        )


class RecordingAnalysisRecordProvider:
    def __init__(self, status: AnalysisRecordCacheStatus) -> None:
        self.status = status
        self.requests: list[AnalysisRecordRequest] = []
        self.iterations = 0

    @contextmanager
    def __call__(
        self, request: AnalysisRecordRequest
    ) -> Iterator[AnalysisRecordSource]:
        self.requests.append(request)

        def records():
            self.iterations += 1
            yield NLPAnalysisRecord(
                chunk_index=0,
                sentence_index=0,
                token_index=0,
                global_token_index=0,
                char_start_in_chunk=0,
                char_end_in_chunk=len(request.text),
                char_start_in_text=0,
                char_end_in_text=len(request.text),
                sentence=request.text,
                token="Rosa",
                lemma="rosa",
                upos="NOUN",
            )

        yield AnalysisRecordSource(
            records=records(),
            cache_status=self.status,
            cache_key="fake-cache-key",
        )


def _backend_factory(config):
    return BuiltNLPBackend(
        backend=FakeBackend(),
        info=NLPBackendInfo(name="fake", language=config.language),
    )


def _base_dependencies():
    return runner_dependencies(
        lambda _path: ensure_app_config(
            {"groups": {"text": {"files": ["input/*.txt"]}}}
        ),
        _backend_factory,
    )


def test_apply_lemma_normalization_is_pure() -> None:
    counter = Counter({"omninus": 2, "omnino": 1})

    result = apply_lemma_normalization(
        counter,
        {"omninus": "omnino"},
    )

    assert result == Counter({"omnino": 3})
    assert counter == Counter({"omninus": 2, "omnino": 1})


def test_split_known_unknown() -> None:
    split = classify_dictionary_entries(
        Counter({"arma": 2, "ignotus": 1}),
        {"arma"},
    )

    assert split.known == Counter({"arma": 2})
    assert split.unknown == Counter({"ignotus": 1})


def test_prepare_run_context_resolves_per_file_work_items(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "sample_text_a.txt").write_text("a", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy", encoding="utf-8")

    def load_config(_path: Path):
        return ensure_app_config({
            "out_dir": "output",
            "groups": {"group_a": {"files": ["input/*.txt"]}},
        })

    deps = runner_dependencies(load_config, _backend_factory)

    context = prepare_run_context(
        corpus_request(tmp_path, config_path, group_by_file=True),
        dependencies=deps,
    )

    assert context.session.corpus.plan.definition.per_file is True
    assert [item.label for item in context.session.corpus.plan.work_items] == ["sample_text_a"]
    assert context.session.corpus.plan.definition.out_dir == (tmp_path / "output").resolve()


def test_analyze_corpora_writes_expected_outputs_from_record_pipeline(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    input_path = tmp_path / "input" / "a.txt"
    input_path.write_text("ignored", encoding="utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "roman.txt").write_text("xiv\n", encoding="utf-8")
    (tmp_path / "config" / "wordlist.txt").write_text("item_a\n", encoding="utf-8")
    (tmp_path / "config" / "ref_tags.txt").write_text("REF\n", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy", encoding="utf-8")

    def load_config(_path: Path):
        return ensure_app_config({
            "out_dir": "output",
            "groups": {"group_a": {"files": ["input/a.txt"]}},
            "filters": {
                "min_token_length": 3,
                "drop_roman_numerals": True,
                "roman_exceptions_file": "config/roman.txt",
            },
            "dictcheck": {
                "enabled": True,
                "wordlist": "config/wordlist.txt",
            },
            "ref_tags": {"enabled": True, "patterns": "config/ref_tags.txt"},
            "trace": {
                "enabled": True,
                "path": "output/trace.tsv",
                "max_rows": 2,
            },
        })

    deps = runner_dependencies(load_config, _backend_factory)
    context = prepare_run_context(
        corpus_request(tmp_path, config_path),
        dependencies=deps,
    )
    corpus = PreparedCorpus(
        label="group_a",
        files=(input_path.resolve(),),
        raw_text="raw",
        prepared_text="prepared text",
        ref_tag_counts=Counter({"tag_a": 1}),
    )

    result = analyze_corpora(
        replace(
            context,
            session=replace(
                context.session,
                corpus=replace(context.session.corpus, corpora=(corpus,)),
            ),
        ),
        analysis_records=deps.analysis_records,
        publication=deps.publication,
    ).groups["group_a"]

    assert result.counter == Counter({"item_a": 2, "item_b": 1})
    generated_names = [artifact.path.name for artifact in context.artifact_plan.artifacts
                       if artifact.group == "group_a" and artifact.kind.value != "diagnostic_trace"]
    assert generated_names == [
        "frequency_group_a.csv",
        "frequency_group_a.known.csv",
        "frequency_group_a.unknown.csv",
        "ref_tags_group_a.csv",
    ]


def test_count_passes_prepared_text_unchanged_to_its_only_nlp_backend(
    tmp_path: Path,
) -> None:
    text = "Rosa amat.\n\n  Marcus venit."
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text(text, encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy", encoding="utf-8")
    backend = FakeNLPBackend()
    factory_calls = []

    def load_config(_path: Path):
        return ensure_app_config({
            "out_dir": "output",
            "groups": {"group_a": {"files": ["input/a.txt"]}},
            "normalization": {"enabled": False},
        })

    def recording_factory(config):
        factory_calls.append(config)
        return BuiltNLPBackend(
            backend=backend,
            info=NLPBackendInfo(name="fake", language=config.language),
        )

    publication = recording_publication_dependencies()
    dependencies = replace(
        runner_dependencies(load_config, recording_factory),
        publication=publication,
    )
    context = prepare_run_context(
        corpus_request(tmp_path, config_path),
        dependencies=dependencies,
    )
    prepared_text = context.session.corpus.corpora[0].prepared_text

    analyze_corpora(
        context,
        analysis_records=dependencies.analysis_records,
        publication=dependencies.publication,
    )

    assert prepared_text == text
    assert backend.calls == [prepared_text]
    assert len(factory_calls) == 1
    group_publisher = cast(
        RecordingGroupArtifactPublisher, publication.group_artifacts
    )
    record_factory = cast(
        InMemoryRecordArtifactSessionFactory, publication.record_artifacts
    )
    assert len(group_publisher.calls) == 1
    assert len(record_factory.requests) == 1
    assert record_factory.sessions[0].exited_normally is True
    assert record_factory.sessions[0].exited_with_exception is False
    assert [record.token for record in record_factory.sessions[0].records] == [
        "Rosa",
        "amat.",
        "Marcus",
        "venit.",
    ]


@pytest.mark.parametrize(
    "status,enabled,expected",
    (
        ("hit", True, (1, 0, 1, 0)),
        ("miss", True, (0, 1, 0, 1)),
        ("disabled", False, (0, 0, 0, 0)),
    ),
)
def test_application_uses_typed_record_provider_and_tracks_cache_status(
    tmp_path: Path,
    status: AnalysisRecordCacheStatus,
    enabled: bool,
    expected: tuple[int, int, int, int],
) -> None:
    text = "Rosa amat.\n\n  Marcus venit."
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text(text, encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy", encoding="utf-8")

    def load_config(_path: Path):
        return ensure_app_config(
            {
                "groups": {"group_a": {"files": ["input/a.txt"]}},
                "normalization": {"enabled": False},
                "analysis_cache": {
                    "enabled": enabled,
                    "dir": "cache/analysis",
                    "lock_timeout_sec": 12.5,
                },
            }
        )

    dependencies = runner_dependencies(load_config, _backend_factory)
    provider = RecordingAnalysisRecordProvider(status)
    dependencies = replace(dependencies, analysis_records=provider)
    context = prepare_run_context(
        corpus_request(tmp_path, config_path), dependencies=dependencies
    )
    result = analyze_corpora(
        context,
        analysis_records=dependencies.analysis_records,
        publication=recording_publication_dependencies(),
    )

    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.text == context.session.corpus.corpora[0].prepared_text == text
    assert request.backend is context.session.backend
    assert request.extraction_policy is context.session.extraction_policy
    assert request.cache.enabled is enabled
    assert request.cache.directory == (tmp_path / "cache/analysis").resolve()
    assert request.cache.lock_timeout_sec == 12.5
    assert provider.iterations == 1
    assert (
        result.cache_stats.hits,
        result.cache_stats.misses,
        result.cache_stats.records_read,
        result.cache_stats.records_written,
    ) == expected
    assert result.cache_stats.groups[0].status == status


def test_summary_lines_and_metadata_include_existing_fields(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("a", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy", encoding="utf-8")

    def load_config(_path: Path):
        return ensure_app_config({
            "out_dir": "output",
            "groups": {"group_a": {"files": ["input/a.txt"]}},
        })

    deps = runner_dependencies(load_config, _backend_factory)
    context = prepare_run_context(
        corpus_request(tmp_path, config_path),
        dependencies=deps,
    )
    analysis = AnalysisResults.from_groups(
        (("group_a", GroupAnalysisResult(
            files=((tmp_path / "input" / "a.txt").resolve(),),
            counter=Counter(),
            ref_tag_counts=Counter(),
        )),),
        cache_stats=AnalysisCacheStatsCollector(enabled=False, directory="").snapshot(),
    )
    partitions = PartitionValidationRunResult((), 0)
    comparisons = ConfiguredComparisonsRunResult(())

    lines = render_run_summary(
        context=context,
        analysis=analysis,
        partitions=partitions,
        comparisons=comparisons,
    )
    meta = run_metadata_to_json_value(build_run_metadata(
        context=context,
        analysis=analysis,
        partitions=partitions,
        comparisons=comparisons,
        environment=collect_runtime_environment(tmp_path),
    ))

    assert lines.splitlines()[:6] == [
        "# Summary",
        "",
        "language: la",
        "stanza_package: perseus",
        "nlp_backend: fake",
        "analysis_unit: lemma",
    ]
    assert meta["groups_files"] == {
        "group_a": [str((tmp_path / "input" / "a.txt").resolve())]
    }
    assert meta["grouping"] == {"mode": "groups"}
    assert meta["nlp"]["backend"] == "fake"
    assert [Path(p).name for p in meta["generated_outputs"]] == [
        "frequency_group_a.csv",
        "summary.txt",
        "run_meta.json",
    ]
