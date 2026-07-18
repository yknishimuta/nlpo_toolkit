from pathlib import Path

from nlpo_toolkit.latin.latin_wordlist import cli
from nlpo_toolkit.latin.latin_wordlist.errors import (
    LatinWordlistConfigError,
    LatinWordlistSourceReadError,
)
from nlpo_toolkit.latin.latin_wordlist.models import (
    LatinWordlistBuildResult,
    WordlistBuildStatistics,
    WordlistNotice,
    WordlistNoticeCode,
)


def test_parse_args_accepts_config_and_has_packaged_default() -> None:
    assert cli.parse_args(["--config", "custom.yml"]).config == Path("custom.yml")
    assert cli.parse_args([]).config == cli.DEFAULT_CONFIG_PATH


def test_cli_success_renders_statistics_and_notices(monkeypatch, capsys) -> None:
    result = LatinWordlistBuildResult(
        output_path=Path("/out/words.txt"),
        word_count=4,
        statistics=WordlistBuildStatistics(1, 2, 1, 1, 1, {}, 4),
        notices=(
            WordlistNotice(
                WordlistNoticeCode.MISSING_EXTRA_WORDLIST,
                Path("missing"),
                "Extra wordlist not found: missing",
            ),
        ),
    )
    monkeypatch.setattr(cli, "load_wordlist_build_request", lambda path: object())
    monkeypatch.setattr(cli, "default_latin_wordlist_dependencies", lambda: object())
    monkeypatch.setattr(cli, "execute_latin_wordlist_build", lambda *args, **kwargs: result)
    assert cli.run_cli(["--config", "config.yml"]) == 0
    captured = capsys.readouterr()
    assert "lemmas from conllu" in captured.out
    assert "wrote 4 words" in captured.out
    assert "[WARN]" in captured.err


def test_cli_maps_config_and_execution_errors(monkeypatch, capsys) -> None:
    def bad_config(path):
        raise LatinWordlistConfigError("bad config")

    monkeypatch.setattr(cli, "load_wordlist_build_request", bad_config)
    assert cli.run_cli([]) == 2
    assert "bad config" in capsys.readouterr().err

    monkeypatch.setattr(cli, "load_wordlist_build_request", lambda path: object())
    monkeypatch.setattr(cli, "default_latin_wordlist_dependencies", lambda: object())

    def bad_source(*args, **kwargs):
        raise LatinWordlistSourceReadError("bad source")

    monkeypatch.setattr(cli, "execute_latin_wordlist_build", bad_source)
    assert cli.run_cli([]) == 1
    assert "bad source" in capsys.readouterr().err
