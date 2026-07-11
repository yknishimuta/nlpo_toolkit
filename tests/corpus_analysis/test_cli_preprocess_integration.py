from __future__ import annotations

from pathlib import Path
import csv

import pytest

from nlpo_toolkit.corpus_analysis import cli
from nlpo_toolkit.corpus_analysis.cli import count as mod
from tests.corpus_analysis.fake_nlp import fake_backend_factory

# Dummy stanza-like objects
class DummySentence:
    def __init__(self, text: str):
        self.text = text

class DummyDoc:
    def __init__(self, sentences):
        self.sentences = sentences

class DummySplitter:
    def __call__(self, text: str):
        return DummyDoc([DummySentence("Puella rosam amat."), DummySentence("Rosa pulchra est.")])


def test_preprocess_cleaner_integration_fixed(tmp_path, monkeypatch):
    """
    Preprocess integration test (cleaner), fixed layout version.

    Verifies:
      - preprocess.kind=cleaner triggers cleaner runner
      - cleaned output dir inferred from cleaner config's 'output'
      - {cleaned_dir} placeholder expands in group globs
      - CSV + summary.txt are created
    """

    # --- Arrange: fake "script directory" so main() can find config/groups.config.yml
    script_dir = tmp_path / "runner_dir"
    (script_dir / "config").mkdir(parents=True, exist_ok=True)

    # cleaner config path (passed to cleaner main)
    cleaner_cfg_path = script_dir / "cleaner.yml"
    cleaner_cfg_path.write_text(
        "\n".join(
            [
                "kind: corpus_corporum",
                "input: input",
                "output: cleaned",
                "",
            ]
        ),
        encoding="utf-8",
    )

    out_dir = script_dir / "output"
    groups_cfg_path = script_dir / "config" / "groups.config.yml"
    groups_cfg_path.write_text(
        "\n".join(
            [
                "preprocess:",
                "  kind: cleaner",
                f"  config: {cleaner_cfg_path}",
                "groups:",
                "  text:",
                '    files: ["{cleaned_dir}/*.txt"]',
                f"out_dir: {out_dir}",
                "language: la",
                "stanza_package: perseus",
                "cpu_only: true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    
    # --- Stub cleaner runner: it should be invoked with argv=[<cleaner_cfg_path>]
    cleaner_called = {"ok": False}

    def fake_cleaner_main(argv):
        assert argv and Path(argv[0]).resolve() == cleaner_cfg_path.resolve()
        cleaner_called["ok"] = True

        cleaned_dir = script_dir / "cleaned"
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        (cleaned_dir / "c1.txt").write_text("Puella rosam amat.\n", encoding="utf-8")
        (cleaned_dir / "c2.txt").write_text("Rosa pulchra est.\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(mod.clean_mod, "main", fake_cleaner_main)

    monkeypatch.setattr(mod, "build_sentence_splitter", lambda *a, **k: DummySplitter())
    monkeypatch.setattr(
        mod,
        "create_nlp_backend",
        fake_backend_factory(
            [("rosa", "rosa", "NOUN"), ("rosa", "rosa", "NOUN"), ("puella", "puella", "NOUN")]
        ),
    )
    monkeypatch.setattr(mod, "render_stanza_package_table", lambda *a, **k: ["[stanza stub]"])

    # --- Act
    rc = cli.main(["count-vocabula", "--project-root", str(script_dir)])
    assert rc == 0
    assert cleaner_called["ok"] is True

    # --- Assert outputs
    csv_path = out_dir / "frequency_text.csv"
    summary_path = out_dir / "summary.txt"
    assert csv_path.exists()
    assert summary_path.exists()

    rows = list(csv.reader(csv_path.open(encoding="utf-8")))
    assert rows[0] == ["lemma", "count"]
    assert len(rows) >= 2
