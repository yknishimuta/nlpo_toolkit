from __future__ import annotations

from collections import Counter
from pathlib import Path

from nlpo_toolkit.backends import BuiltNLPBackend, NLPBackendInfo
from nlpo_toolkit.corpus_analysis.analysis_pipeline import (
    analyze_one_corpus,
    apply_lemma_normalization,
    split_known_unknown,
)
from nlpo_toolkit.corpus_analysis.corpus import PreparedCorpus
from nlpo_toolkit.corpus_analysis.run_reporting import (
    build_final_run_metadata,
    build_summary_lines,
)
from nlpo_toolkit.corpus_analysis.runner_types import (
    AnalysisResults,
    ComparisonRunResult,
    PartitionRunResult,
    RunnerDependencies,
)
from nlpo_toolkit.corpus_analysis.runtime import prepare_run_context
from nlpo_toolkit.models import NLPDocument, NLPSentence, NLPToken


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
    return RunnerDependencies(
        load_config=lambda _path: {},
        clean_module=object(),
        backend_factory=_backend_factory,
        render_stanza_package_table=lambda *_args, **_kwargs: [],
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
    known, unknown = split_known_unknown(
        Counter({"arma": 2, "ignotus": 1}),
        {"arma"},
    )

    assert known == Counter({"arma": 2})
    assert unknown == Counter({"ignotus": 1})


def test_prepare_run_context_resolves_per_file_work_items(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "sample_text_a.txt").write_text("a", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy", encoding="utf-8")

    def load_config(_path: Path):
        return {
            "out_dir": "output",
            "groups": {"group_a": {"files": ["input/*.txt"]}},
        }

    deps = RunnerDependencies(
        load_config=load_config,
        clean_module=object(),
        backend_factory=_backend_factory,
        render_stanza_package_table=lambda *_args, **_kwargs: [],
    )

    context = prepare_run_context(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=True,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        dependencies=deps,
    )

    assert context.plan.per_file is True
    assert [item.label for item in context.plan.work_items] == ["sample_text_a"]
    assert context.plan.out_dir == (tmp_path / "output").resolve()


def test_analyze_one_corpus_writes_expected_outputs_from_record_pipeline(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    input_path = tmp_path / "input" / "a.txt"
    input_path.write_text("ignored", encoding="utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "roman.txt").write_text("xiv\n", encoding="utf-8")
    (tmp_path / "config" / "wordlist.txt").write_text("item_a\n", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy", encoding="utf-8")

    def load_config(_path: Path):
        return {
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
            "ref_tags": {"enabled": True},
            "trace": {
                "enabled": True,
                "path": "output/trace.tsv",
                "max_rows": 2,
            },
        }

    deps = RunnerDependencies(
        load_config=load_config,
        clean_module=object(),
        backend_factory=_backend_factory,
        render_stanza_package_table=lambda *_args, **_kwargs: [],
    )
    context = prepare_run_context(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=False,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        dependencies=deps,
    )
    corpus = PreparedCorpus(
        label="group_a",
        files=(input_path.resolve(),),
        raw_text="raw",
        prepared_text="prepared text",
        ref_tag_counts=Counter({"tag_a": 1}),
    )

    result = analyze_one_corpus(
        context=context,
        dependencies=deps,
        corpus=corpus,
        trace_paths={"group_a": tmp_path / "output" / "trace_group_a.tsv"},
        lemma_normalization_map=None,
    )

    assert result.counter == Counter({"item_a": 2, "item_b": 1})
    generated_names = [path.name for path in result.generated_outputs]
    assert generated_names == [
        "ref_tags_group_a.csv",
        "frequency_group_a.csv",
        "frequency_group_a.known.csv",
        "frequency_group_a.unknown.csv",
    ]


def test_summary_lines_and_metadata_include_existing_fields(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("a", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy", encoding="utf-8")

    def load_config(_path: Path):
        return {
            "out_dir": "output",
            "groups": {"group_a": {"files": ["input/a.txt"]}},
        }

    deps = RunnerDependencies(
        load_config=load_config,
        clean_module=object(),
        backend_factory=_backend_factory,
        render_stanza_package_table=lambda *_args, **_kwargs: [],
    )
    context = prepare_run_context(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=False,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        dependencies=deps,
    )
    analysis = AnalysisResults(
        groups=(),
        counters_by_group={},
        files_by_group={"group_a": ((tmp_path / "input" / "a.txt").resolve(),)},
        ref_tags_by_group={},
        trace_paths={},
        generated_outputs=(),
    )
    partitions = PartitionRunResult((), (), (), (), 0)
    comparisons = ComparisonRunResult((), (), ())

    lines = build_summary_lines(
        context=context,
        analysis=analysis,
        partitions=partitions,
        comparisons=comparisons,
        dependencies=deps,
    )
    meta = build_final_run_metadata(
        context=context,
        analysis=analysis,
        partitions=partitions,
        comparisons=comparisons,
        generated_outputs=(context.plan.out_dir / "summary.txt", context.plan.out_dir / "run_meta.json"),
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
        "summary.txt",
        "run_meta.json",
    ]
