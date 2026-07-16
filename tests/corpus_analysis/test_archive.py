import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.archive.errors import RunArchiveError
from nlpo_toolkit.corpus_analysis.archive.file_metadata import file_sha256
from nlpo_toolkit.corpus_analysis.archive.models import ArchivedFile
from nlpo_toolkit.corpus_analysis.archive.service import create_run_archive
from nlpo_toolkit.corpus_analysis.archive_types import (
    ArchivedFileCounts,
    RunArchiveRequest,
)
from nlpo_toolkit.corpus_analysis.config import ensure_app_config
from nlpo_toolkit.corpus_analysis.config_references import (
    ConfigFileReference,
    ConfigArchivePolicy,
    ResolvedConfigFiles,
)
from nlpo_toolkit.corpus_analysis.count_command import CountRequest, execute_count_command
from nlpo_toolkit.corpus_analysis.requests import CorpusPreparationRequest
from nlpo_toolkit.corpus_analysis.composition import default_runner_dependencies
from nlpo_toolkit.corpus_analysis.ports import CountCommandDependencies
from nlpo_toolkit.corpus_analysis.run_plan import AnalysisPlan, ResolvedAnalysisPlan
from nlpo_toolkit.corpus_analysis.runner_types import RunResult
from nlpo_toolkit.corpus_analysis.artifacts.models import (
    ArtifactKind, ArtifactPlan, PlannedArtifact,
)


def make_run_result(
    tmp_path: Path,
    *,
    output_files: tuple[Path, ...] = (),
    trace_files: tuple[Path, ...] = (),
    input_files: tuple[Path, ...] = (),
    cleaned_files: tuple[Path, ...] = (),
    config_references: tuple[ConfigFileReference, ...] = (),
) -> RunResult:
    config_path = tmp_path / "config.yml"
    config_path.write_text("groups: {g: {files: []}}\n", encoding="utf-8")
    config = ensure_app_config({"groups": {"g": {"files": []}}})
    definition = AnalysisPlan(
        project_root=tmp_path.resolve(),
        config_path=config_path.resolve(),
        config=config,
        grouping_mode="groups",
        error_on_empty_group=False,
        cleaner_plan=None,
        config_files=ResolvedConfigFiles(config_references),
    )
    plan = ResolvedAnalysisPlan(
        definition=definition,
        cleaned_dir=(tmp_path / "cleaned").resolve(),
        work_items=(),
        group_files={"g": tuple(input_files)},
    )
    artifacts = []
    for path in output_files:
        kind = (ArtifactKind.RUN_METADATA if path.name == "run_meta.json"
                else ArtifactKind.SUMMARY if path.name == "summary.txt"
                else ArtifactKind.FREQUENCY)
        artifacts.append(PlannedArtifact(
            kind, path, group="g" if kind is ArtifactKind.FREQUENCY else None
        ))
    artifacts.extend(
        PlannedArtifact(ArtifactKind.DIAGNOSTIC_TRACE, path, group=f"trace-{index}")
        for index, path in enumerate(trace_files)
    )
    return RunResult(
        exit_code=0,
        plan=plan,
        groups_files={"g": tuple(input_files or cleaned_files)},
        input_files=tuple(input_files),
        cleaned_files=tuple(cleaned_files),
        artifact_plan=ArtifactPlan(tuple(artifacts)),
        config_references=plan.config_files.references,
    )


def test_archive_dtos_reject_ambiguous_or_invalid_values(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        RunArchiveRequest(archive_root="runs")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ArchivedFileCounts(inputs=-1)
    source = tmp_path / "source.txt"
    source.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        ArchivedFile(
            source_path=source.resolve(),
            archive_relative_path=Path("../escape.txt"),
            sha256="0" * 64,
            size_bytes=1,
        )


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

    archive = create_run_archive(run_result=result, request=RunArchiveRequest(run_name="exact"))

    assert (archive.archive_directory / "outputs" / declared.name).exists()
    assert (archive.archive_directory / "outputs" / run_meta.name).read_text() == run_meta.read_text()
    assert not (archive.archive_directory / "outputs" / stale.name).exists()


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
        config_references=(
            ConfigFileReference(
                kind="root_config",
                source_path=config.resolve(),
                archive_policy=ConfigArchivePolicy.SNAPSHOT,
                snapshot_relative_path=Path("config/root.yml"),
            ),
            ConfigFileReference(
                kind="dictcheck.wordlist",
                source_path=wordlist.resolve(),
                archive_policy=ConfigArchivePolicy.METADATA_ONLY,
            ),
        ),
    )

    archive = create_run_archive(
        run_result=result,
        request=RunArchiveRequest(run_name="inventory", include_cleaned_files=True),
    )

    assert (archive.archive_directory / "trace" / "exact.tsv").exists()
    assert (archive.archive_directory / "cleaned" / "selected.txt").exists()
    assert not (archive.archive_directory / "cleaned" / "stale.txt").exists()
    manifest = json.loads((archive.archive_directory / "manifest.json").read_text())
    assert manifest["external_references"][0]["sha256"] == file_sha256(wordlist)
    assert manifest["groups_files"] == {"g": [str(selected)]}


def test_archive_options_counts_timestamp_and_existing_directory(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("x", encoding="utf-8")
    result = make_run_result(tmp_path, input_files=(source,))
    request = RunArchiveRequest(run_name=None, include_input_files=True, created_at=datetime(2026, 7, 6, 12, 34, 56, tzinfo=timezone.utc))
    archive = create_run_archive(run_result=result, request=request)
    assert archive.archive_directory.name == "20260706-123456"
    assert archive.copied_files.inputs == 1
    with pytest.raises(RunArchiveError, match="already exists"):
        create_run_archive(run_result=result, request=request)


def test_missing_declared_output_fails_without_partial_archive(tmp_path: Path) -> None:
    result = make_run_result(tmp_path, output_files=(tmp_path / "missing.csv",))
    with pytest.raises(RunArchiveError, match="Archive source file.*run output"):
        create_run_archive(run_result=result, request=RunArchiveRequest(run_name="missing"))
    assert not (tmp_path / "runs" / "missing").exists()


def test_count_cli_uses_run_result_without_reloading_config_or_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = make_run_result(tmp_path)
    dependencies = CountCommandDependencies(
        runner=default_runner_dependencies(),
        run_analysis=lambda *_args, **_kwargs: result,
        archive_creator=create_run_archive,
    )
    command_result = execute_count_command(
        CountRequest(
            corpus=CorpusPreparationRequest(tmp_path, result.plan.config_path),
            run_name="cli-result",
        ),
        dependencies=dependencies,
    )

    assert command_result.successful is True
    assert command_result.archive is not None
    assert (tmp_path / "runs" / "cli-result" / "manifest.json").exists()
    assert capsys.readouterr().out == ""


def test_count_cli_does_not_archive_nonzero_run_result(
    tmp_path: Path,
) -> None:
    result = replace(make_run_result(tmp_path), exit_code=1)
    dependencies = CountCommandDependencies(
        runner=default_runner_dependencies(),
        run_analysis=lambda *_args, **_kwargs: result,
        archive_creator=create_run_archive,
    )
    command_result = execute_count_command(
        CountRequest(
            corpus=CorpusPreparationRequest(tmp_path, result.plan.config_path),
            run_name="must-not-exist",
        ),
        dependencies=dependencies,
    )

    assert command_result.successful is False
    assert command_result.archive is None
    assert not (tmp_path / "runs" / "must-not-exist").exists()
