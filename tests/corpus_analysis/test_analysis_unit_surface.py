import csv
from pathlib import Path

from nlpo_toolkit.corpus_analysis.runner import run
from tests.corpus_analysis.fake_nlp import (
    corpus_request,
    fake_backend_factory,
    runner_dependencies,
)


def test_analysis_unit_surface_writes_word_frequency_and_passes_use_lemma_false(tmp_path):
    # --- Arrange: fake "repo" layout that main() expects ---
    script_dir = tmp_path / "runner_dir"
    (script_dir / "config").mkdir(parents=True, exist_ok=True)
    config_path = script_dir / "config" / "groups.config.yml"
    config_path.write_text("dummy", encoding="utf-8")

    # Create a real input file to match glob
    data_dir = script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "t1.txt").write_text("Puella rosam amat.\n", encoding="utf-8")

    out_dir = script_dir / "output"

    cfg = {
        "groups": {"text": {"files": [str(data_dir / "*.txt")]}},
        "out_dir": str(out_dir),
        "analysis_unit": "surface",
        "nlp": {
            "language": "la",
            "stanza_package": "perseus",
            "cpu_only": True,
        },
        "dictcheck": {"enabled": False},
    }

    def fake_load_config(path: Path):
        assert path.name == "groups.config.yml"
        return cfg


    dependencies = runner_dependencies(
        fake_load_config,
        fake_backend_factory(
            [("Rosa", "lemma_rosa", "NOUN"), ("puella", "lemma_puella", "NOUN")]
        ),
    )
    # --- Act ---
    result = run(
        corpus_request(script_dir, config_path),
        dependencies=dependencies,
    )
    assert result.exit_code == 0

    # --- Assert: header changes for surface ---
    csv_path = out_dir / "frequency_text.csv"
    assert csv_path.exists()

    rows = list(csv.reader(csv_path.open(encoding="utf-8")))
    assert rows[0] == ["word", "frequency"]
    assert {row[0] for row in rows[1:]} == {"rosa", "puella"}
