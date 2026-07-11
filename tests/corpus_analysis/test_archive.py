import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.archive import (
    ArchiveOptions,
    RunArchiveError,
    create_run_archive,
    file_sha256,
    sanitize_run_name,
)
from nlpo_toolkit.corpus_analysis.config import ensure_app_config
from nlpo_toolkit.corpus_analysis.run_plan import RunPlan
from nlpo_toolkit.corpus_analysis.runner_types import ReferencedConfigFile, RunResult
from nlpo_toolkit.corpus_analysis.cli import count as count_cli


def make_run_result(
    tmp_path: Path,
    *,
    output_files: tuple[Path, ...] = (),
    trace_files: tuple[Path, ...] = (),
    input_files: tuple[Path, ...] = (),
    cleaned_files: tuple[Path, ...] = (),
    config_files: tuple[ReferencedConfigFile, ...] = (),
) -> RunResult:
    config_path = tmp_path / "config.yml"
    config_path.write_text("groups: {g: {files: []}}\n", encoding="utf-8")
    config = ensure_app_config({"groups": {"g": {"files": []}}})
    plan = RunPlan(
        project_root=tmp_path.resolve(), config_path=config_path.resolve(), config=config,
        out_dir=(tmp_path / "output").resolve(), cleaned_dir=(tmp_path / "cleaned").resolve(),
        grouping_mode="groups", per_file=False, auto_mode=False, auto_group_name="text",
        work_items=(), group_files={"g": tuple(input_files)}, partition_specs=(), comparison_specs=(),
        analysis_unit="lemma", use_lemma=True, csv_header=("lemma", "count"),
    )
    summary = next((p for p in output_files if p.name == "summary.txt"), tmp_path / "summary.txt")
    metadata = next((p for p in output_files if p.name == "run_meta.json"), tmp_path / "run_meta.json")
    return RunResult(0, plan, {"g": tuple(input_files or cleaned_files)}, tuple(input_files), tuple(cleaned_files), tuple(output_files), tuple(trace_files), tuple(config_files), summary, metadata)


def test_sanitize_run_name() -> None:
    assert sanitize_run_name("virgil noun 01") == "virgil_noun_01"
    with pytest.raises(ValueError):
        sanitize_run_name("../bad")


def test_archive_copies_only_declared_outputs_and_does_not_parse_run_meta(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    declared = output / "frequency_text.csv"
    stale = output / "stale.csv"
    run_meta = output / "run_meta.json"
    declared.write_text("item,count\nrosa,1\n", encoding="utf-8")
    stale.write_text("old,99\n", encoding="utf-8")
    run_meta.write_text('{not valid json; "generated_outputs": ["stale.csv"]', encoding="utf-8")
    result = make_run_result(tmp_path, output_files=(declared, run_meta))

    archive = create_run_archive(result=result, options=ArchiveOptions(run_name="exact"))

    assert (archive.run_dir / "outputs" / declared.name).exists()
    assert (archive.run_dir / "outputs" / run_meta.name).read_text() == run_meta.read_text()
    assert not (archive.run_dir / "outputs" / stale.name).exists()


def test_archive_uses_exact_cleaned_trace_config_and_external_inventory(tmp_path: Path) -> None:
    cleaned_dir = tmp_path / "cleaned"
    cleaned_dir.mkdir()
    selected = cleaned_dir / "selected.txt"
    stale = cleaned_dir / "stale.txt"
    selected.write_text("selected", encoding="utf-8")
    stale.write_text("stale", encoding="utf-8")
    trace = tmp_path / "custom" / "exact.tsv"
    trace.parent.mkdir()
    trace.write_text("trace", encoding="utf-8")
    config = tmp_path / "root.yml"
    config.write_text("changed groups: stale", encoding="utf-8")
    wordlist = tmp_path / "words.txt"
    wordlist.write_text("rosa", encoding="utf-8")
    result = make_run_result(
        tmp_path, trace_files=(trace,), cleaned_files=(selected,),
        config_files=(ReferencedConfigFile("root_config", config, True, Path("config/root.yml")), ReferencedConfigFile("dictcheck.wordlist", wordlist, False)),
    )

    archive = create_run_archive(result=result, options=ArchiveOptions(run_name="inventory", include_cleaned=True))

    assert (archive.run_dir / "trace" / "exact.tsv").exists()
    assert (archive.run_dir / "cleaned" / "selected.txt").exists()
    assert not (archive.run_dir / "cleaned" / "stale.txt").exists()
    manifest = json.loads((archive.run_dir / "manifest.json").read_text())
    assert manifest["external_references"][0]["sha256"] == file_sha256(wordlist)
    assert manifest["groups_files"] == {"g": [str(selected)]}


def test_archive_options_counts_timestamp_and_existing_directory(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("x", encoding="utf-8")
    result = make_run_result(tmp_path, input_files=(source,))
    options = ArchiveOptions(run_name=None, include_input=True, created_at=datetime(2026, 7, 6, 12, 34, 56, tzinfo=timezone.utc))
    archive = create_run_archive(result=result, options=options)
    assert archive.run_dir.name == "20260706-123456"
    assert archive.copied_input_count == 1
    with pytest.raises(RunArchiveError, match="already exists"):
        create_run_archive(result=result, options=options)


def test_missing_declared_output_fails_without_partial_archive(tmp_path: Path) -> None:
    result = make_run_result(tmp_path, output_files=(tmp_path / "missing.csv",))
    with pytest.raises(RunArchiveError, match="Declared run output"):
        create_run_archive(result=result, options=ArchiveOptions(run_name="missing"))
    assert not (tmp_path / "runs" / "missing").exists()


def test_count_cli_uses_run_result_without_reloading_config_or_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = make_run_result(tmp_path)
    monkeypatch.setattr(count_cli, "run", lambda **kwargs: result)
    monkeypatch.setattr(
        count_cli,
        "load_config",
        lambda _path: (_ for _ in ()).throw(AssertionError("config reloaded")),
    )

    rc = count_cli.run_count_vocabula(
        project_root=tmp_path,
        config_path=result.plan.config_path,
        run_name="cli-result",
    )

    assert rc == 0
    assert (tmp_path / "runs" / "cli-result" / "manifest.json").exists()
    assert "included input files: 0" in capsys.readouterr().out


def test_count_cli_does_not_archive_nonzero_run_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = replace(make_run_result(tmp_path), exit_code=1)
    monkeypatch.setattr(count_cli, "run", lambda **kwargs: result)

    rc = count_cli.run_count_vocabula(
        project_root=tmp_path,
        config_path=result.plan.config_path,
        run_name="must-not-exist",
    )

    assert rc == 1
    assert not (tmp_path / "runs" / "must-not-exist").exists()
