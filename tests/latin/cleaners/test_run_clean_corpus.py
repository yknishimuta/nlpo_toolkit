from pathlib import Path

import pytest

import nlpo_toolkit.latin.cleaners.run_clean_corpus as cli
from nlpo_toolkit.cleaner_contracts import (
    CleanerApplicationError,
    CleanerConfig,
    CleanerConfigInspection,
    CleanerExecutionResult,
)


def _inspection(path: Path) -> CleanerConfigInspection:
    config = CleanerConfig(path, "scholastic_text", path.parent / "in.txt", path.parent / "out")
    return CleanerConfigInspection(config, (config.input_path,), ())


def test_cli_builds_typed_request_and_presents_result(tmp_path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "cleaner.yml"
    inspection = _inspection(config_path)
    result = CleanerExecutionResult(
        config_path, "scholastic_text", tmp_path / "out", (), tmp_path / "events.tsv"
    )
    calls = []
    monkeypatch.setattr(cli, "inspect_cleaner_config", lambda path: inspection)
    monkeypatch.setattr(cli, "execute_cleaner", lambda request: calls.append(request) or result)

    assert cli.main([str(config_path)]) == 0
    assert calls[0].inspection is inspection
    output = capsys.readouterr().out
    assert "scholastic_text" in output
    assert "events.tsv" in output


def test_cli_converts_domain_error_to_exit_code(tmp_path, monkeypatch, capsys) -> None:
    failure = CleanerApplicationError("broken cleaner")
    monkeypatch.setattr(cli, "inspect_cleaner_config", lambda path: (_ for _ in ()).throw(failure))

    assert cli.main([str(tmp_path / "config.yml")]) == 1
    assert "[ERROR] broken cleaner" in capsys.readouterr().err


def test_cli_rejects_extra_arguments() -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["one.yml", "two.yml"])
    assert caught.value.code == 2
