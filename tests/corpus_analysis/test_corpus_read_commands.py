from __future__ import annotations

from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis import cli


def _write_broken_config(tmp_path: Path) -> tuple[Path, Path]:
    broken = tmp_path / "broken.txt"
    broken.write_bytes(b"\xff\xfe\xfa")
    config = tmp_path / "groups.yml"
    config.write_text(
        "\n".join(
            (
                "groups:",
                "  broken:",
                "    files:",
                "      - broken.txt",
                "out_dir: output",
                "",
            )
        ),
        encoding="utf-8",
    )
    return config, broken


@pytest.mark.parametrize(
    ("command", "output_flag", "output_name"),
    (
        ("count", None, None),
        ("features", "--out", "features.csv"),
        ("ngram", "--out", "ngrams.tsv"),
    ),
)
def test_commands_fail_without_partial_outputs_on_corpus_read_error(
    tmp_path: Path,
    capsys,
    command: str,
    output_flag: str | None,
    output_name: str | None,
) -> None:
    config, broken = _write_broken_config(tmp_path)
    args = [
        command,
        "--project-root",
        str(tmp_path),
        "--config",
        str(config),
    ]
    if command == "ngram":
        args.extend(("--field", "token"))
    if output_flag is not None and output_name is not None:
        args.extend((output_flag, str(tmp_path / output_name)))

    rc = cli.main(args)

    assert rc == 1
    assert broken.name in capsys.readouterr().err
    assert not (tmp_path / "output").exists()
    if output_name is not None:
        assert not (tmp_path / output_name).exists()
