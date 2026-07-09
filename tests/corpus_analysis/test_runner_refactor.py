from __future__ import annotations

from collections import Counter
from pathlib import Path

from nlpo_toolkit.backends import BuiltNLPBackend, NLPBackendInfo
from nlpo_toolkit.corpus_analysis.corpus import PreparedCorpus
import nlpo_toolkit.corpus_analysis.runner as runner_mod
from nlpo_toolkit.models import NLPDocument


class FakeBackend:
    def __call__(self, text: str) -> NLPDocument:
        return NLPDocument(text=text)


def _backend_factory(config):
    return BuiltNLPBackend(
        backend=FakeBackend(),
        info=NLPBackendInfo(name="fake", language=config.language),
    )


def _base_dependencies(*, count_group=None):
    return runner_mod.RunnerDependencies(
        load_config=lambda _path: {},
        clean_module=object(),
        backend_factory=_backend_factory,
        count_group=count_group or (lambda *_args, **_kwargs: Counter()),
        render_stanza_package_table=lambda *_args, **_kwargs: [],
    )


def test_apply_lemma_normalization_is_pure() -> None:
    counter = Counter({"omninus": 2, "omnino": 1})

    result = runner_mod.apply_lemma_normalization(
        counter,
        {"omninus": "omnino"},
    )

    assert result == Counter({"omnino": 3})
    assert counter == Counter({"omninus": 2, "omnino": 1})


def test_split_known_unknown() -> None:
    known, unknown = runner_mod.split_known_unknown(
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

    deps = runner_mod.RunnerDependencies(
        load_config=load_config,
        clean_module=object(),
        backend_factory=_backend_factory,
        count_group=lambda *_args, **_kwargs: Counter(),
        render_stanza_package_table=lambda *_args, **_kwargs: [],
    )

    context = runner_mod.prepare_run_context(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=True,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        dependencies=deps,
    )

    assert context.per_file is True
    assert [item.label for item in context.work_items] == ["sample_text_a"]
    assert context.out_dir == (tmp_path / "output").resolve()


def test_analyze_one_corpus_writes_expected_outputs_and_passes_filter_args(tmp_path: Path) -> None:
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

    captured: dict[str, object] = {}

    def count_group(text, nlp, **kwargs):
        captured["text"] = text
        captured.update(kwargs)
        return Counter({"item_a": 2, "item_b": 1})

    deps = runner_mod.RunnerDependencies(
        load_config=load_config,
        clean_module=object(),
        backend_factory=_backend_factory,
        count_group=count_group,
        render_stanza_package_table=lambda *_args, **_kwargs: [],
    )
    context = runner_mod.prepare_run_context(
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

    result = runner_mod.analyze_one_corpus(
        context=context,
        dependencies=deps,
        corpus=corpus,
        trace_paths={"group_a": tmp_path / "output" / "trace_group_a.tsv"},
        lemma_normalization_map=None,
    )

    assert captured["text"] == "prepared text"
    assert captured["use_lemma"] is True
    assert captured["min_token_length"] == 3
    assert captured["drop_roman_numerals"] is True
    assert captured["roman_exceptions"] == frozenset({"xiv"})
    assert captured["trace_max_rows"] == 2
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

    deps = runner_mod.RunnerDependencies(
        load_config=load_config,
        clean_module=object(),
        backend_factory=_backend_factory,
        count_group=lambda *_args, **_kwargs: Counter(),
        render_stanza_package_table=lambda *_args, **_kwargs: [],
    )
    context = runner_mod.prepare_run_context(
        project_root=tmp_path,
        script_dir=None,
        config_path=config_path,
        group_by_file=False,
        auto_single_cleaned=False,
        error_on_empty_group=False,
        dependencies=deps,
    )
    analysis = runner_mod.AnalysisResults(
        groups=(),
        counters_by_group={},
        files_by_group={"group_a": ((tmp_path / "input" / "a.txt").resolve(),)},
        ref_tags_by_group={},
        trace_paths={},
        generated_outputs=(),
    )
    partitions = runner_mod.PartitionRunResult((), (), (), (), 0)
    comparisons = runner_mod.ComparisonRunResult((), (), ())

    lines = runner_mod.build_summary_lines(
        context=context,
        analysis=analysis,
        partitions=partitions,
        comparisons=comparisons,
        dependencies=deps,
    )
    meta = runner_mod.build_final_run_metadata(
        context=context,
        analysis=analysis,
        partitions=partitions,
        comparisons=comparisons,
        generated_outputs=(context.out_dir / "summary.txt", context.out_dir / "run_meta.json"),
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
