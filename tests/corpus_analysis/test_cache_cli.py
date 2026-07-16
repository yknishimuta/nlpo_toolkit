from __future__ import annotations

from nlpo_toolkit.corpus_analysis import cli
from nlpo_toolkit.corpus_analysis.cache import CacheClearError, resolve_cache_dir
import pytest
import nlpo_toolkit.corpus_analysis.cache as cache_module


def test_cache_clear_uses_default_cache_dir_without_config(tmp_path, capsys):
    cache_dir = tmp_path / ".analysis_cache"
    cache_dir.mkdir()
    (cache_dir / "entry.json").write_text("{}", encoding="utf-8")

    rc = cli.main(["cache", "clear", "--project-root", str(tmp_path)])

    assert rc == 0
    assert not cache_dir.exists()
    assert "[OK] cache cleared: .analysis_cache" in capsys.readouterr().out


def test_cache_clear_reads_default_config_when_present(tmp_path, capsys):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "groups.config.yml").write_text(
        "analysis_cache:\n  dir: cache/analysis\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache" / "analysis"
    cache_dir.mkdir(parents=True)
    (cache_dir / "entry.json").write_text("{}", encoding="utf-8")

    rc = cli.main(["cache", "clear", "--project-root", str(tmp_path)])

    assert rc == 0
    assert not cache_dir.exists()
    assert "[OK] cache cleared: cache/analysis" in capsys.readouterr().out


def test_cache_clear_uses_explicit_config_relative_to_project_root(tmp_path):
    (tmp_path / "custom").mkdir()
    (tmp_path / "custom" / "groups.yml").write_text(
        "analysis_cache:\n  dir: custom-cache\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / "custom-cache"
    cache_dir.mkdir()

    rc = cli.main(
        [
            "cache",
            "clear",
            "--project-root",
            str(tmp_path),
            "--config",
            "custom/groups.yml",
        ]
    )

    assert rc == 0
    assert not cache_dir.exists()


def test_cache_clear_missing_cache_is_ok(tmp_path, capsys):
    rc = cli.main(["cache", "clear", "--project-root", str(tmp_path)])

    assert rc == 0
    assert "[OK] cache already clean: .analysis_cache" in capsys.readouterr().out


def test_cache_clear_rejects_cache_dir_outside_project_root(tmp_path, capsys):
    outside = tmp_path.parent / "outside-cache"
    outside.mkdir(exist_ok=True)
    (outside / "entry.json").write_text("{}", encoding="utf-8")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "groups.config.yml").write_text(
        "analysis_cache:\n  dir: ../outside-cache\n",
        encoding="utf-8",
    )

    rc = cli.main(["cache", "clear", "--project-root", str(tmp_path)])

    assert rc == 1
    assert outside.exists()
    assert "Refusing to clear cache outside project root" in capsys.readouterr().err


def test_resolve_cache_dir_rejects_project_root(tmp_path):
    config_path = tmp_path / "groups.yml"
    config_path.write_text("analysis_cache:\n  dir: .\n", encoding="utf-8")

    try:
        resolve_cache_dir(tmp_path, config_path)
    except CacheClearError as exc:
        assert "project root" in str(exc)
    else:
        raise AssertionError("Expected CacheClearError")


def test_cache_clear_unlinks_cache_file(tmp_path):
    cache_path = tmp_path / ".analysis_cache"
    cache_path.write_text("cache", encoding="utf-8")

    rc = cli.main(["cache", "clear", "--project-root", str(tmp_path)])

    assert rc == 0
    assert not cache_path.exists()


def test_cache_clear_does_not_use_removed_cache_key(tmp_path, capsys):
    removed_key = "lemma" + "_cache"
    config_path = tmp_path / "groups.yml"
    config_path.write_text(f"{removed_key}:\n  dir: cache/old\n", encoding="utf-8")
    old_cache_dir = tmp_path / "cache" / "old"
    old_cache_dir.mkdir(parents=True)
    default_cache_dir = tmp_path / ".analysis_cache"
    default_cache_dir.mkdir()

    rc = cli.main(
        [
            "cache",
            "clear",
            "--project-root",
            str(tmp_path),
            "--config",
            str(config_path),
        ]
    )

    assert rc == 0
    assert old_cache_dir.exists()
    assert not default_cache_dir.exists()
    assert "[OK] cache cleared: .analysis_cache" in capsys.readouterr().out


@pytest.mark.parametrize("section", (
    "analysis_cache:\n  dir: 12\n",
    "analysis_cache:\n  unknown: value\n",
    "analysis_cache:\n  dir: first\n  dir: second\n",
))
def test_cache_config_errors_do_not_delete_cache(tmp_path, section) -> None:
    config = tmp_path / "groups.yml"
    config.write_text(section, encoding="utf-8")
    cache_dir = tmp_path / ".analysis_cache"
    cache_dir.mkdir()
    (cache_dir / "entry").write_text("data", encoding="utf-8")
    with pytest.raises(CacheClearError):
        resolve_cache_dir(tmp_path, config)
    assert cache_dir.exists()


def test_cache_config_is_loaded_once(tmp_path, monkeypatch) -> None:
    config = tmp_path / "groups.yml"
    config.write_text("analysis_cache:\n  dir: cache\n", encoding="utf-8")
    real_loader = cache_module.load_yaml_mapping
    calls = 0

    def counted(path):
        nonlocal calls
        calls += 1
        return real_loader(path)

    monkeypatch.setattr(cache_module, "load_yaml_mapping", counted)
    assert resolve_cache_dir(tmp_path, config) == (tmp_path / "cache").resolve()
    assert calls == 1
