from __future__ import annotations

from tests.corpus_analysis.fake_nlp import corpus_request

from collections import Counter
from dataclasses import replace
from pathlib import Path

from nlpo_toolkit.nlp.contracts import BuiltNLPBackend, NLPBackendInfo
from nlpo_toolkit.corpus_analysis.analysis_orchestration import analyze_corpora
from nlpo_toolkit.corpus_analysis.analysis_cache.stats import AnalysisCacheRunStats
from nlpo_toolkit.corpus_analysis.analysis_results import AnalysisResults, GroupAnalysisResult
from nlpo_toolkit.corpus_analysis.analysis_outputs import (
    apply_lemma_normalization,
    split_known_unknown,
)
from nlpo_toolkit.corpus_analysis.corpus import PreparedCorpus
from nlpo_toolkit.corpus_analysis.config import ensure_app_config
from nlpo_toolkit.corpus_analysis.run_reporting import (
    build_final_run_metadata,
    build_summary_lines,
)
from nlpo_toolkit.corpus_analysis.runner_types import (
    ComparisonRunResult,
    PartitionRunResult,
)
from nlpo_toolkit.corpus_analysis.runtime import prepare_run_context
from nlpo_toolkit.nlp.contracts import NLPDocument, NLPSentence, NLPToken
from tests.corpus_analysis.fake_nlp import runner_dependencies


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
    split = split_known_unknown(
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
        )
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
        cache_stats=AnalysisCacheRunStats(enabled=False, directory=""),
    )
    partitions = PartitionRunResult((), (), (), 0)
    comparisons = ComparisonRunResult((), ())

    lines = build_summary_lines(
        context=context,
        analysis=analysis,
        partitions=partitions,
        comparisons=comparisons,
    )
    meta = build_final_run_metadata(
        context=context,
        analysis=analysis,
        partitions=partitions,
        comparisons=comparisons,
    )

    assert lines[:6] == [
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
