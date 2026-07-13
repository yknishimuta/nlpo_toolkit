from __future__ import annotations

from pathlib import Path

from nlpo_toolkit.corpus_analysis.runner import run
from tests.corpus_analysis.fake_nlp import fake_backend_factory, runner_dependencies


def test_dictcheck_disabled_does_not_create_known_unknown(tmp_path):
    # --- Arrange: fake "repo" layout that main() expects ---
    script_dir = tmp_path / "runner_dir"
    (script_dir / "config").mkdir(parents=True, exist_ok=True)
    config_path = script_dir / "config" / "groups.config.yml"
    config_path.write_text("dummy", encoding="utf-8")

    # Create a real input file to match glob
    data_dir = script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "t1.txt").write_text("Puella rosam amat.\n", encoding="utf-8")
    wordlist = data_dir / "wordlists" / "latin_words.txt"
    wordlist.parent.mkdir()
    wordlist.write_text("rosa\n", encoding="utf-8")

    out_dir = script_dir / "output"

    # Config returned by load_config (new design)
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
            "enabled": False,
            # Explicit paths are validated even when the feature is disabled.
            "wordlist": str(wordlist),
        },
    }

    def fake_load_config(path: Path):
        # main() reads: script_dir/config/groups.config.yml
        assert path.name == "groups.config.yml"
        return cfg

    dependencies = runner_dependencies(
        fake_load_config,
        fake_backend_factory([("rosa", "rosa", "NOUN"), ("rosa", "rosa", "NOUN")]),
    )
    # --- Act ---
    result = run(
        project_root=script_dir,
        config_path=config_path,
        dependencies=dependencies,
    )
    assert result.exit_code == 0

    # --- Assert ---
    # Base CSV should exist
    assert (out_dir / "frequency_text.csv").exists()
    assert (out_dir / "summary.txt").exists()

    # But dictcheck outputs must NOT exist when enabled=False
    assert not (out_dir / "frequency_text.known.csv").exists()
    assert not (out_dir / "frequency_text.unknown.csv").exists()
