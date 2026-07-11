from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

import nlpo_toolkit.corpus_analysis.runner as runner_mod
from tests.corpus_analysis.fake_nlp import FakeNLPBackend, fake_backend_factory


def _backend_for_counters(counter_by_text: dict[str, Counter]) -> FakeNLPBackend:
    return FakeNLPBackend(
        per_text={
            text: tuple((key, key, "NOUN") for key, count in counter.items() for _ in range(count))
            for text, counter in counter_by_text.items()
        }
    )


def _run(
    tmp_path: Path,
    cfg: dict,
    counter_by_text: dict[str, Counter],
    *,
    capsys=None,
) -> int:
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")

    return runner_mod.run(
        project_root=tmp_path,
        config_path=config_path,
        load_config_fn=lambda _p: cfg,
        clean_mod=object(),
        backend_factory=fake_backend_factory(backend=_backend_for_counters(counter_by_text)),
        build_sentence_splitter_fn=None,
        render_stanza_package_table_fn=lambda *a, **k: [],
    )


def _write_inputs(tmp_path: Path, names: list[str]) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in names:
        (input_dir / f"{name}.txt").write_text(name.upper(), encoding="utf-8")


def _base_cfg(*, on_mismatch: str = "error", report: str = "mismatches") -> dict:
    return {
        "out_dir": "output",
        "groups": {
            "full": {"files": ["input/full.txt"]},
            "part_a": {"files": ["input/part_a.txt"]},
            "part_b": {"files": ["input/part_b.txt"]},
        },
        "validations": {
            "partitions": [
                {
                    "name": "full_split",
                    "whole": "full",
                    "parts": ["part_a", "part_b"],
                    "on_mismatch": on_mismatch,
                    "report": report,
                }
            ]
        },
    }


def test_runner_validates_final_group_counters_and_writes_outputs(tmp_path: Path) -> None:
    _write_inputs(tmp_path, ["full", "part_a", "part_b"])
    rc = _run(
        tmp_path,
        _base_cfg(report="all"),
        {
            "FULL": Counter({"a": 3, "b": 2}),
            "PART_A": Counter({"a": 1, "b": 2}),
            "PART_B": Counter({"a": 2}),
        },
    )

    assert rc.exit_code == 0
    out_dir = tmp_path / "output"
    csv_path = out_dir / "partition_validation_full_split.csv"
    json_path = out_dir / "partition_validation.json"
    summary_path = out_dir / "summary.txt"
    meta_path = out_dir / "run_meta.json"
    assert csv_path.exists()
    assert json_path.exists()
    assert summary_path.exists()
    assert meta_path.exists()

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert rows == [
        {
            "item": "a",
            "whole_count": "3",
            "part_a_count": "1",
            "part_b_count": "2",
            "parts_sum": "3",
            "delta": "0",
            "status": "match",
        },
        {
            "item": "b",
            "whole_count": "2",
            "part_a_count": "2",
            "part_b_count": "0",
            "parts_sum": "2",
            "delta": "0",
            "status": "match",
        },
    ]

    summary = summary_path.read_text(encoding="utf-8")
    assert "# Partition validation" in summary
    assert "name=full_split status=OK" in summary

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["partition_validations"][0]["exact_match"] is True
    generated_names = {Path(p).name for p in meta["generated_outputs"]}
    assert "partition_validation_full_split.csv" in generated_names
    assert "partition_validation.json" in generated_names

    partition_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert partition_json["partitions"][0]["csv"] == "partition_validation_full_split.csv"


def test_runner_compares_after_lemma_normalize(tmp_path: Path) -> None:
    _write_inputs(tmp_path, ["full", "part_a", "part_b"])
    norm_path = tmp_path / "lemma_normalize.tsv"
    norm_path.write_text("x\ta\n", encoding="utf-8")
    cfg = _base_cfg()
    cfg["dictcheck"] = {"lemma_normalize": "lemma_normalize.tsv"}

    rc = _run(
        tmp_path,
        cfg,
        {
            "FULL": Counter({"x": 1}),
            "PART_A": Counter({"a": 1}),
            "PART_B": Counter(),
        },
    )

    assert rc.exit_code == 0
    meta = json.loads((tmp_path / "output" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["partition_validations"][0]["exact_match"] is True


def test_runner_warn_mismatch_returns_zero_and_writes_stderr(tmp_path: Path, capsys) -> None:
    _write_inputs(tmp_path, ["full", "part_a", "part_b"])

    rc = _run(
        tmp_path,
        _base_cfg(on_mismatch="warn"),
        {
            "FULL": Counter({"a": 3}),
            "PART_A": Counter({"a": 2}),
            "PART_B": Counter(),
        },
    )

    err = capsys.readouterr().err
    assert rc.exit_code == 0
    assert "[WARN] partition full_split mismatch: token_delta=1 mismatched_items=1" in err


def test_runner_error_mismatch_returns_one_but_keeps_outputs(tmp_path: Path, capsys) -> None:
    _write_inputs(tmp_path, ["full", "part_a", "part_b"])

    rc = _run(
        tmp_path,
        _base_cfg(on_mismatch="error"),
        {
            "FULL": Counter({"a": 3}),
            "PART_A": Counter({"a": 2}),
            "PART_B": Counter(),
        },
    )

    err = capsys.readouterr().err
    assert rc.exit_code == 1
    assert "[ERROR] partition full_split mismatch: token_delta=1 mismatched_items=1" in err
    assert (tmp_path / "output" / "partition_validation_full_split.csv").exists()
    assert (tmp_path / "output" / "partition_validation.json").exists()
    assert (tmp_path / "output" / "run_meta.json").exists()


def test_runner_rejects_empty_partition_reference_even_without_empty_group_flag(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "full.txt").write_text("FULL", encoding="utf-8")
    (input_dir / "part_a.txt").write_text("PART_A", encoding="utf-8")
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")

    with pytest.raises(ValueError, match="Partition full_split references empty group: part_b"):
        runner_mod.run(
            project_root=tmp_path,
            config_path=config_path,
            load_config_fn=lambda _p: _base_cfg(),
            clean_mod=object(),
            backend_factory=fake_backend_factory(),
            build_sentence_splitter_fn=None,
            render_stanza_package_table_fn=lambda *a, **k: [],
        )
