from __future__ import annotations

from pathlib import Path

from nlpo_toolkit.count_vocabula import cli


def test_count_vocabula_cli_uses_project_root_default_config(tmp_path, monkeypatch):
    calls = []

    def fake_run_count_vocabula(**kwargs) -> int:
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(cli, "run_count_vocabula", fake_run_count_vocabula)

    rc = cli.main(["count-vocabula", "--project-root", str(tmp_path)])

    assert rc == 0
    assert calls[0]["project_root"] == tmp_path.resolve()
    assert calls[0]["config_path"] == tmp_path / "config" / "groups.config.yml"
    assert calls[0]["group_by_file"] is False
    assert calls[0]["archive_run"] is False
    assert calls[0]["run_name"] is None


def test_count_alias_accepts_config_relative_to_project_root(tmp_path, monkeypatch):
    calls = []

    def fake_run_count_vocabula(**kwargs) -> int:
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(cli, "run_count_vocabula", fake_run_count_vocabula)

    rc = cli.main(["count", "--project-root", str(tmp_path), "--config", "custom.yml"])

    assert rc == 0
    assert calls[0]["project_root"] == tmp_path.resolve()
    assert calls[0]["config_path"] == (tmp_path / "custom.yml").resolve()
    assert calls[0]["group_by_file"] is False


def test_count_vocabula_cli_accepts_group_by_file(tmp_path, monkeypatch):
    calls = []

    def fake_run_count_vocabula(**kwargs) -> int:
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(cli, "run_count_vocabula", fake_run_count_vocabula)

    rc = cli.main(["count-vocabula", "--project-root", str(tmp_path), "--group-by-file"])

    assert rc == 0
    assert calls[0]["project_root"] == tmp_path.resolve()
    assert calls[0]["config_path"] == tmp_path / "config" / "groups.config.yml"
    assert calls[0]["group_by_file"] is True


def test_count_vocabula_cli_accepts_run_archive_options(tmp_path, monkeypatch):
    calls = []

    def fake_run_count_vocabula(**kwargs) -> int:
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(cli, "run_count_vocabula", fake_run_count_vocabula)

    rc = cli.main(
        [
            "count-vocabula",
            "--project-root",
            str(tmp_path),
            "--run-name",
            "my run",
            "--runs-dir",
            "archives",
            "--include-cleaned",
            "--include-input",
        ]
    )

    assert rc == 0
    assert calls[0]["archive_run"] is False
    assert calls[0]["run_name"] == "my run"
    assert calls[0]["runs_dir"] == Path("archives")
    assert calls[0]["include_cleaned"] is True
    assert calls[0]["include_input"] is True
    assert calls[0]["command_line"][0:2] == ["nlpo", "count-vocabula"]


def test_count_vocabula_cli_accepts_dry_run(tmp_path, monkeypatch):
    calls = []

    def fake_dry_run_count_vocabula(**kwargs) -> int:
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(cli, "dry_run_count_vocabula", fake_dry_run_count_vocabula)

    rc = cli.main(["count-vocabula", "--dry-run", "--project-root", str(tmp_path), "--group-by-file"])

    assert rc == 0
    assert calls[0]["project_root"] == tmp_path.resolve()
    assert calls[0]["config_path"] == tmp_path / "config" / "groups.config.yml"
    assert calls[0]["group_by_file"] is True
    assert calls[0]["error_on_empty_group"] is False
