from __future__ import annotations

from tests.corpus_analysis.fake_nlp import corpus_request

import csv
import json
from pathlib import Path

from nlpo_toolkit.corpus_analysis.config import load_config
from nlpo_toolkit.corpus_analysis.archive.service import create_run_archive
from nlpo_toolkit.corpus_analysis.archive.contracts import RunArchiveRequest
from nlpo_toolkit.corpus_analysis import runner as runner_mod
from tests.corpus_analysis.fake_nlp import fake_backend_factory, runner_dependencies


def _run(project_root: Path, config_path: Path, *, group_by_file: bool = False) -> int:
    return runner_mod.run(
        corpus_request(project_root, config_path, group_by_file=group_by_file),
        dependencies=runner_dependencies(load_config, fake_backend_factory()),
    )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_multiple_groups_get_separate_trace_files_and_labels(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("item_a", encoding="utf-8")
    (tmp_path / "input" / "b.txt").write_text("item_b", encoding="utf-8")
    config_path = tmp_path / "groups.config.yml"
    config_path.write_text(
        "\n".join(
            [
                "groups:",
                "  group_a: {files: [input/a.txt]}",
                "  group_b: {files: [input/b.txt]}",
                "trace:",
                "  enabled: true",
                "  path: output/trace.tsv",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = _run(tmp_path, config_path)
    assert result.exit_code == 0
    trace_a = tmp_path / "output" / "trace_group_a.tsv"
    trace_b = tmp_path / "output" / "trace_group_b.tsv"

    assert trace_a.exists()
    assert trace_b.exists()
    assert "group_a" in trace_a.read_text(encoding="utf-8")
    assert "group_b" not in trace_a.read_text(encoding="utf-8")
    assert "group_b" in trace_b.read_text(encoding="utf-8")
    assert "group_a" not in trace_b.read_text(encoding="utf-8")
    assert _rows(trace_a)[0]["label"] == "group_a"
    assert _rows(trace_b)[0]["label"] == "group_b"

    meta = json.loads((tmp_path / "output" / "run_meta.json").read_text(encoding="utf-8"))
    assert set(meta["trace"]["files"]) == {"group_a", "group_b"}


def test_single_group_uses_base_trace_path(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "only.txt").write_text("item_a", encoding="utf-8")
    config_path = tmp_path / "groups.config.yml"
    config_path.write_text(
        "groups:\n  only_group: {files: [input/only.txt]}\ntrace:\n  enabled: true\n  path: output/trace.tsv\n",
        encoding="utf-8",
    )

    assert _run(tmp_path, config_path).exit_code == 0
    assert (tmp_path / "output" / "trace.tsv").exists()
    assert not (tmp_path / "output" / "trace_only_group.tsv").exists()
    assert _rows(tmp_path / "output" / "trace.tsv")[0]["label"] == "only_group"


def test_group_by_file_uses_unique_file_labels_for_trace(tmp_path: Path) -> None:
    (tmp_path / "input" / "one").mkdir(parents=True)
    (tmp_path / "input" / "two").mkdir(parents=True)
    (tmp_path / "input" / "one" / "text.txt").write_text("item_a", encoding="utf-8")
    (tmp_path / "input" / "two" / "text.txt").write_text("item_b", encoding="utf-8")
    config_path = tmp_path / "groups.config.yml"
    config_path.write_text(
        "groups:\n  all: {files: [input/**/*.txt]}\ntrace:\n  enabled: true\n  path: output/trace.tsv\n",
        encoding="utf-8",
    )

    assert _run(tmp_path, config_path, group_by_file=True).exit_code == 0
    assert (tmp_path / "output" / "trace_text.tsv").exists()
    assert (tmp_path / "output" / "trace_text_2.tsv").exists()
    assert _rows(tmp_path / "output" / "trace_text.tsv")[0]["label"] == "text"
    assert _rows(tmp_path / "output" / "trace_text_2.tsv")[0]["label"] == "text_2"


def test_custom_trace_path_parent_and_suffix_are_preserved(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("item_a", encoding="utf-8")
    (tmp_path / "input" / "b.txt").write_text("item_b", encoding="utf-8")
    config_path = tmp_path / "groups.config.yml"
    config_path.write_text(
        "\n".join(
            [
                "groups:",
                "  group_a: {files: [input/a.txt]}",
                "  group_b: {files: [input/b.txt]}",
                "trace:",
                "  enabled: true",
                "  path: artifacts/custom-trace.tsv",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert _run(tmp_path, config_path).exit_code == 0
    assert (tmp_path / "artifacts" / "custom-trace_group_a.tsv").exists()
    assert (tmp_path / "artifacts" / "custom-trace_group_b.tsv").exists()
    assert not (tmp_path / "output" / "custom-trace_group_a.tsv").exists()


def test_trace_disabled_writes_no_trace_files(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("item_a", encoding="utf-8")
    config_path = tmp_path / "groups.config.yml"
    config_path.write_text(
        "groups:\n  group_a: {files: [input/a.txt]}\ntrace:\n  enabled: false\n  path: output/trace.tsv\n",
        encoding="utf-8",
    )

    assert _run(tmp_path, config_path).exit_code == 0
    assert not list((tmp_path / "output").glob("trace*.tsv"))


def test_trace_max_rows_applies_per_trace_file(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("item_a", encoding="utf-8")
    (tmp_path / "input" / "b.txt").write_text("item_b", encoding="utf-8")
    config_path = tmp_path / "groups.config.yml"
    config_path.write_text(
        "\n".join(
            [
                "groups:",
                "  group_a: {files: [input/a.txt]}",
                "  group_b: {files: [input/b.txt]}",
                "trace:",
                "  enabled: true",
                "  path: output/trace.tsv",
                "  max_rows: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert _run(tmp_path, config_path).exit_code == 0
    assert len(_rows(tmp_path / "output" / "trace_group_a.tsv")) == 1
    assert len(_rows(tmp_path / "output" / "trace_group_b.tsv")) == 1
    assert _rows(tmp_path / "output" / "trace_group_b.tsv")[0]["label"] == "group_b"


def test_archive_includes_all_trace_files_from_run_meta(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("item_a", encoding="utf-8")
    (tmp_path / "input" / "b.txt").write_text("item_b", encoding="utf-8")
    config_path = tmp_path / "groups.config.yml"
    config_path.write_text(
        "\n".join(
            [
                "groups:",
                "  group_a: {files: [input/a.txt]}",
                "  group_b: {files: [input/b.txt]}",
                "trace:",
                "  enabled: true",
                "  path: output/trace.tsv",
                "",
            ]
        ),
        encoding="utf-8",
    )
    result = _run(tmp_path, config_path)
    assert result.exit_code == 0

    archive = create_run_archive(run_result=result, request=RunArchiveRequest(run_name="trace_run"))
    run_dir = archive.archive_directory

    copied = {p.name for p in (run_dir / "trace").iterdir()}
    assert copied == {"trace_group_a.tsv", "trace_group_b.tsv"}
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_names = {Path(row["archive_path"]).name for row in manifest["trace_files"]}
    assert manifest_names == {"trace_group_a.tsv", "trace_group_b.tsv"}
