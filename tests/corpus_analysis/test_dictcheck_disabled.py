from __future__ import annotations

from pathlib import Path

from nlpo_toolkit.corpus_analysis import cli
from nlpo_toolkit.corpus_analysis.cli import count as mod
from tests.corpus_analysis.fake_nlp import fake_backend_factory, runner_dependencies


def test_dictcheck_disabled_does_not_create_known_unknown(tmp_path, monkeypatch):
    # --- Arrange: fake "repo" layout that main() expects ---
    script_dir = tmp_path / "runner_dir"
    (script_dir / "config").mkdir(parents=True, exist_ok=True)

    # Create a real input file to match glob
    data_dir = script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "t1.txt").write_text("Puella rosam amat.\n", encoding="utf-8")

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
            # even if present, must not be used when enabled=False
            "wordlist": str(script_dir / "data" / "wordlists" / "latin_words.txt"),
        },
    }

    def fake_load_config(path: Path):
        # main() reads: script_dir/config/groups.config.yml
        assert path.name == "groups.config.yml"
        return cfg

    # Make main() think config exists
    real_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if self.name == "groups.config.yml":
            return True
        return real_exists(self)

    monkeypatch.setattr(mod.Path, "exists", fake_exists)

    
    dependencies = runner_dependencies(
        fake_load_config,
        fake_backend_factory([("rosa", "rosa", "NOUN"), ("rosa", "rosa", "NOUN")]),
    )
    monkeypatch.setattr(mod, "default_runner_dependencies", lambda: dependencies)

    # --- Act ---
    rc = cli.main(["count", "--project-root", str(script_dir)])
    assert rc == 0

    # --- Assert ---
    # Base CSV should exist
    assert (out_dir / "frequency_text.csv").exists()
    assert (out_dir / "summary.txt").exists()

    # But dictcheck outputs must NOT exist when enabled=False
    assert not (out_dir / "frequency_text.known.csv").exists()
    assert not (out_dir / "frequency_text.unknown.csv").exists()
