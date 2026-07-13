from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nlpo_toolkit.corpus_analysis import cli
from nlpo_toolkit.corpus_analysis.cli import count as count_cli
from nlpo_toolkit.corpus_analysis.count_command import CountRequest
from nlpo_toolkit.corpus_analysis.dependencies import CountCommandDependencies


def _capture_count_request(monkeypatch, *, exit_code: int = 0):
    calls: list[tuple[CountRequest, CountCommandDependencies]] = []

    def fake_execute_count_command(
        request: CountRequest,
        *,
        dependencies: CountCommandDependencies,
    ):
        calls.append((request, dependencies))
        return SimpleNamespace(
            successful=exit_code == 0,
            archive=None,
            run=SimpleNamespace(partition_mismatches=()),
        )

    monkeypatch.setattr(
        count_cli,
        "execute_count_command",
        fake_execute_count_command,
    )
    return calls


def test_count_cli_builds_canonical_request(tmp_path, monkeypatch) -> None:
    calls = _capture_count_request(monkeypatch, exit_code=7)

    rc = cli.main(
        [
            "count",
            "--project-root",
            str(tmp_path),
            "--config",
            "custom.yml",
            "--group-by-file",
            "--run-name",
            "my run",
            "--runs-dir",
            "archives",
            "--include-cleaned",
            "--include-input",
            "--error-on-empty-group",
        ]
    )

    assert rc == 1
    request, dependencies = calls[0]
    assert request.corpus.project_root == tmp_path.resolve()
    assert request.corpus.config_path == (tmp_path / "custom.yml").resolve()
    assert request.corpus.grouping_override == "per_file"
    assert request.run_name == "my run"
    assert request.runs_dir == Path("archives")
    assert request.include_cleaned is True
    assert request.include_input is True
    assert request.corpus.error_on_empty_group is True
    assert request.command_line[:2] == ("nlpo", "count")
    assert isinstance(dependencies, CountCommandDependencies)


def test_count_cli_default_request(tmp_path, monkeypatch) -> None:
    calls = _capture_count_request(monkeypatch)

    rc = cli.main(
        ["count", "--project-root", str(tmp_path)]
    )

    assert rc == 0
    request, _dependencies = calls[0]
    assert request.corpus.config_path == tmp_path / "config" / "groups.config.yml"
    assert request.corpus.grouping_override is None
    assert request.archive_run is False


def test_count_cli_grouping_overrides_are_mutually_exclusive(tmp_path) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(
            [
                "count",
                "--project-root",
                str(tmp_path),
                "--group-by-file",
                "--auto-single-cleaned",
            ]
        )
    assert caught.value.code == 2
