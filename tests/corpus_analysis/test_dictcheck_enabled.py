from __future__ import annotations

from pathlib import Path
import csv

import pytest

from nlpo_toolkit.corpus_analysis.runner import run
from tests.corpus_analysis.fake_nlp import fake_backend_factory, runner_dependencies


def test_dictcheck_enabled_creates_known_unknown(tmp_path):
    # --- Arrange: fake "repo" layout that main() expects ---
    script_dir = tmp_path / "runner_dir"
    (script_dir / "config").mkdir(parents=True, exist_ok=True)

    # Create a real input file to match glob
    data_dir = script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "t1.txt").write_text("Puella rosam amat.\n", encoding="utf-8")

    out_dir = script_dir / "output"

    # Wordlist (relative path case): data/wordlists/latin_words.txt
    wordlist_dir = data_dir / "wordlists"
    wordlist_dir.mkdir(parents=True, exist_ok=True)
    wordlist_path = wordlist_dir / "latin_words.txt"
    wordlist_path.write_text(
        "\n".join(["rosa"]),  # only "rosa" is known
        encoding="utf-8",
    )

    cfg = {
        "groups": {
            "text": {"files": [str(data_dir / "*.txt")]}
        },
        "out_dir": str(out_dir),
        "nlp": {
            "language": "la",
            "stanza_package": "perseus",
            "cpu_only": True,
        },
        "dictcheck": {
            "enabled": True,
            # IMPORTANT: main() resolves relative to script_dir
            "wordlist": "data/wordlists/latin_words.txt",
            # optional: you can omit these because we default to word/frequency in the helper
            # "lemma_column": "word",
            # "count_column": "frequency",
        },
    }

    config_path = script_dir / "config" / "groups.config.yml"
    config_path.write_text("dummy", encoding="utf-8")

    def fake_load_config(path: Path):
        assert path.name == "groups.config.yml"
        return cfg

    dependencies = runner_dependencies(
        fake_load_config,
        fake_backend_factory(
            [("rosa", "rosa", "NOUN"), ("rosa", "rosa", "NOUN"), ("puella", "puella", "NOUN")]
        ),
    )
    result = run(
        project_root=script_dir,
        config_path=config_path,
        dependencies=dependencies,
    )
    assert result.exit_code == 0

    # --- Assert ---
    base_csv = out_dir / "frequency_text.csv"
    known_csv = out_dir / "frequency_text.known.csv"
    unknown_csv = out_dir / "frequency_text.unknown.csv"

    assert base_csv.exists()
    assert known_csv.exists()
    assert unknown_csv.exists()

    # base csv header is ["lemma","count"]
    rows = list(csv.reader(base_csv.open(encoding="utf-8")))
    assert rows[0] == ["lemma", "count"]

    # known should contain rosa, unknown should contain puella
    known_rows = list(csv.DictReader(known_csv.open(encoding="utf-8")))
    unknown_rows = list(csv.DictReader(unknown_csv.open(encoding="utf-8")))

    assert [r["lemma"] for r in known_rows] == ["rosa"]
    assert [r["lemma"] for r in unknown_rows] == ["puella"]


def test_dictcheck_enabled_requires_wordlist(tmp_path):
    script_dir = tmp_path / "runner_dir"
    (script_dir / "config").mkdir(parents=True, exist_ok=True)

    data_dir = script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "t1.txt").write_text("Puella rosam amat.\n", encoding="utf-8")

    out_dir = script_dir / "output"

    cfg = {
        "groups": {"text": {"files": [str(data_dir / "*.txt")]}},
        "out_dir": str(out_dir),
        "nlp": {
            "language": "la",
            "stanza_package": "perseus",
            "cpu_only": True,
        },
        "dictcheck": {
            "enabled": True,
            # wordlist missing on purpose
        },
    }
    config_path = script_dir / "config" / "groups.config.yml"
    config_path.write_text("dummy", encoding="utf-8")

    def fake_load_config(path: Path):
        assert path.name == "groups.config.yml"
        return cfg

    dependencies = runner_dependencies(
        fake_load_config,
        fake_backend_factory([("rosa", "rosa", "NOUN")]),
    )
    with pytest.raises(ValueError, match=r"dictcheck\.wordlist"):
        run(
            project_root=script_dir,
            config_path=config_path,
            dependencies=dependencies,
        )
