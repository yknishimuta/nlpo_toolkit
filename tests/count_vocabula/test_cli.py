from __future__ import annotations

from pathlib import Path

from nlpo_toolkit.count_vocabula import cli


def test_count_vocabula_cli_uses_project_root_default_config(tmp_path, monkeypatch):
    calls = []

    def fake_run_count_vocabula(*, project_root: Path, config_path: Path) -> int:
        calls.append((project_root, config_path))
        return 0

    monkeypatch.setattr(cli, "run_count_vocabula", fake_run_count_vocabula)

    rc = cli.main(["count-vocabula", "--project-root", str(tmp_path)])

    assert rc == 0
    assert calls == [(tmp_path.resolve(), tmp_path / "config" / "groups.config.yml")]


def test_count_alias_accepts_config_relative_to_project_root(tmp_path, monkeypatch):
    calls = []

    def fake_run_count_vocabula(*, project_root: Path, config_path: Path) -> int:
        calls.append((project_root, config_path))
        return 0

    monkeypatch.setattr(cli, "run_count_vocabula", fake_run_count_vocabula)

    rc = cli.main(["count", "--project-root", str(tmp_path), "--config", "custom.yml"])

    assert rc == 0
    assert calls == [(tmp_path.resolve(), (tmp_path / "custom.yml").resolve())]
