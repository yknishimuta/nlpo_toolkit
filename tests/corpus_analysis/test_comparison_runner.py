from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

import nlpo_toolkit.corpus_analysis.runner as runner_mod
from nlpo_toolkit.corpus_analysis.archive import ArchiveOptions, create_run_archive
from nlpo_toolkit.corpus_analysis.count_command import CountRequest
from nlpo_toolkit.corpus_analysis.dependencies import CorpusPlanningDependencies
from nlpo_toolkit.corpus_analysis.dry_run import execute_dry_run
from nlpo_toolkit.corpus_analysis.config import load_config
from tests.corpus_analysis.fake_nlp import FakeNLPBackend, fake_backend_factory, runner_dependencies


def _write_inputs(project_root: Path) -> None:
    (project_root / "input").mkdir()
    (project_root / "input" / "corpus_a.txt").write_text("sample_text_a", encoding="utf-8")
    (project_root / "input" / "corpus_b.txt").write_text("sample_text_b", encoding="utf-8")


def _run_with_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cfg: dict,
    *,
    counter_by_text: dict[str, Counter[str]] | None = None,
) -> int:
    project_root = tmp_path
    config_path = project_root / "groups.config.yml"
    config_path.write_text("dummy", encoding="utf-8")


    if counter_by_text is None:
        counter_by_text = {
            "sample_text_a": Counter({"item_common": 1}),
            "sample_text_b": Counter({"item_common": 1}),
        }
    backend = FakeNLPBackend(
        per_text={
            text: tuple((key, key, "NOUN") for key, count in counter.items() for _ in range(count))
            for text, counter in counter_by_text.items()
        }
    )

    return runner_mod.run(
        project_root=project_root,
        config_path=config_path,
        dependencies=runner_dependencies(
            lambda _p: cfg,
            fake_backend_factory(backend=backend),
        ),
    )


def test_runner_generates_comparison_outputs_summary_meta_and_uses_final_counter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path
    _write_inputs(project_root)
    (project_root / "config").mkdir()
    normalize_path = project_root / "config" / "lemma_normalize.tsv"
    normalize_path.write_text("item_source\titem_a\n", encoding="utf-8")

    cfg = {
        "out_dir": "output",
        "analysis_unit": "lemma",
        "groups": {
            "corpus_a": {"files": ["input/corpus_a.txt"]},
            "corpus_b": {"files": ["input/corpus_b.txt"]},
        },
        "dictcheck": {"lemma_normalize": "config/lemma_normalize.tsv"},
        "comparisons": [
            {
                "name": "comparison_1",
                "group_a": "corpus_a",
                "group_b": "corpus_b",
                "scale": 10000,
                "zero_correction": 0.5,
                "min_total_count": 1,
            }
        ],
    }

    rc = _run_with_config(
        project_root,
        monkeypatch,
        cfg,
        counter_by_text={
            "sample_text_a": Counter({"item_source": 2, "item_common": 1}),
            "sample_text_b": Counter({"item_b": 2, "item_common": 1}),
        },
    )

    assert rc.exit_code == 0
    out_dir = project_root / "output"
    csv_path = out_dir / "group_comparison_comparison_1.csv"
    json_path = out_dir / "group_comparisons.json"
    assert csv_path.exists()
    assert json_path.exists()

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    items = {row["item"] for row in rows}
    assert "item_a" in items
    assert "item_source" not in items
    row_a = next(row for row in rows if row["item"] == "item_a")
    assert row_a["comparison"] == "comparison_1"
    assert row_a["analysis_unit"] == "lemma"
    assert row_a["group_a"] == "corpus_a"
    assert row_a["group_b"] == "corpus_b"
    assert row_a["group_a_count"] == "2"
    assert row_a["group_b_count"] == "0"
    assert row_a["direction"] == "corpus_a"

    summary = (out_dir / "summary.txt").read_text(encoding="utf-8")
    assert "# Group comparisons" in summary
    assert "name=comparison_1 group_a=corpus_a group_b=corpus_b" in summary

    meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
    generated_names = {Path(p).name for p in meta["generated_outputs"]}
    assert "group_comparison_comparison_1.csv" in generated_names
    assert "group_comparisons.json" in generated_names
    assert meta["group_comparisons"][0]["name"] == "comparison_1"
    assert meta["group_comparisons"][0]["csv"] == "group_comparison_comparison_1.csv"

    overview = json.loads(json_path.read_text(encoding="utf-8"))
    assert overview["analysis_unit"] == "lemma"
    assert overview["comparisons"][0]["rows_after_filter"] == 3


def test_runner_surface_mode_generates_generic_item_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_inputs(tmp_path)
    cfg = {
        "out_dir": "output",
        "analysis_unit": "surface",
        "groups": {
            "corpus_a": {"files": ["input/corpus_a.txt"]},
            "corpus_b": {"files": ["input/corpus_b.txt"]},
        },
        "comparisons": [{"name": "comparison_1", "group_a": "corpus_a", "group_b": "corpus_b"}],
    }

    rc = _run_with_config(
        tmp_path,
        monkeypatch,
        cfg,
        counter_by_text={
            "sample_text_a": Counter({"item_a": 1}),
            "sample_text_b": Counter({"item_b": 1}),
        },
    )

    assert rc.exit_code == 0
    with (tmp_path / "output" / "group_comparison_comparison_1.csv").open(encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    assert "item" in header
    assert "analysis_unit" in header


def test_runner_without_comparisons_does_not_generate_comparison_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_inputs(tmp_path)
    cfg = {
        "out_dir": "output",
        "groups": {
            "corpus_a": {"files": ["input/corpus_a.txt"]},
            "corpus_b": {"files": ["input/corpus_b.txt"]},
        },
    }

    rc = _run_with_config(tmp_path, monkeypatch, cfg)

    assert rc.exit_code == 0
    out_dir = tmp_path / "output"
    assert not (out_dir / "group_comparisons.json").exists()
    assert not list(out_dir.glob("group_comparison_*.csv"))


def test_runner_rejects_empty_comparison_counter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_inputs(tmp_path)
    cfg = {
        "out_dir": "output",
        "groups": {
            "corpus_a": {"files": ["input/corpus_a.txt"]},
            "corpus_b": {"files": ["input/corpus_b.txt"]},
        },
        "comparisons": [{"name": "comparison_1", "group_a": "corpus_a", "group_b": "corpus_b"}],
    }

    with pytest.raises(ValueError, match="comparison 'comparison_1': group 'corpus_b' has zero target tokens"):
        _run_with_config(
            tmp_path,
            monkeypatch,
            cfg,
            counter_by_text={
                "sample_text_a": Counter({"item_a": 1}),
                "sample_text_b": Counter(),
            },
        )


def test_runner_rejects_group_by_file_with_comparisons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_inputs(tmp_path)
    config_path = tmp_path / "groups.config.yml"
    config_path.write_text("dummy", encoding="utf-8")
    cfg = {
        "out_dir": "output",
        "groups": {
            "corpus_a": {"files": ["input/corpus_a.txt"]},
            "corpus_b": {"files": ["input/corpus_b.txt"]},
        },
        "comparisons": [{"name": "comparison_1", "group_a": "corpus_a", "group_b": "corpus_b"}],
    }

    with pytest.raises(ValueError, match="comparisons cannot be used"):
        runner_mod.run(
            project_root=tmp_path,
            config_path=config_path,
            group_by_file=True,
            dependencies=runner_dependencies(
                lambda _p: cfg,
                fake_backend_factory([("item_a", "item_a", "NOUN")]),
            ),
        )


def test_dry_run_reports_comparison_and_empty_reference(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project_root = tmp_path
    (project_root / "config").mkdir()
    _write_inputs(project_root)
    config_path = project_root / "config" / "groups.config.yml"
    config_path.write_text(
        "\n".join(
            [
                "groups:",
                "  corpus_a: {files: [input/corpus_a.txt]}",
                "  corpus_b: {files: [input/corpus_b.txt]}",
                "  corpus_c: {files: [input/missing.txt]}",
                "comparisons:",
                "  - {name: comparison_1, group_a: corpus_a, group_b: corpus_b}",
                "  - {name: comparison_2, group_a: corpus_a, group_b: corpus_c}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rc = execute_dry_run(
        request=CountRequest(
            project_root=project_root,
            config_path=config_path,
            dry_run=True,
        ),
        dependencies=CorpusPlanningDependencies(
            load_config=load_config,
            cleaner_loader=lambda: pytest.fail(
                "cleaner loader must not be called"
            ),
        ),
    )

    out = capsys.readouterr().out
    assert rc == 1
    assert "[OK] comparison comparison_1: group_a=corpus_a group_b=corpus_b scale=10000 min_total_count=1" in out
    assert "[ERROR] comparison comparison_2 references empty group: corpus_c" in out


def test_archive_includes_current_comparison_outputs_from_generated_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_inputs(tmp_path)
    cfg = {
        "out_dir": "output",
        "groups": {
            "corpus_a": {"files": ["input/corpus_a.txt"]},
            "corpus_b": {"files": ["input/corpus_b.txt"]},
        },
        "comparisons": [{"name": "comparison_1", "group_a": "corpus_a", "group_b": "corpus_b"}],
    }
    config_path = tmp_path / "groups.config.yml"
    config_path.write_text(
        "\n".join(
            [
                "groups:",
                "  corpus_a: {files: [input/corpus_a.txt]}",
                "  corpus_b: {files: [input/corpus_b.txt]}",
                "comparisons:",
                "  - {name: comparison_1, group_a: corpus_a, group_b: corpus_b}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rc = _run_with_config(
        tmp_path,
        monkeypatch,
        cfg,
        counter_by_text={
            "sample_text_a": Counter({"item_a": 1}),
            "sample_text_b": Counter({"item_b": 1}),
        },
    )
    assert rc.exit_code == 0

    archive = create_run_archive(result=rc, options=ArchiveOptions(run_name="comparison_1"))
    run_dir = archive.run_dir

    copied = {path.name for path in (run_dir / "outputs").iterdir()}
    assert "group_comparison_comparison_1.csv" in copied
    assert "group_comparisons.json" in copied
