import json
from pathlib import Path

from nlpo_toolkit.corpus_analysis.runner import run
from tests.corpus_analysis.fake_nlp import fake_backend_factory, runner_dependencies


def test_normalization_policy_is_written_to_summary_and_run_meta(tmp_path):
    # Arrange
    script_dir = tmp_path / "runner_dir"
    (script_dir / "config").mkdir(parents=True, exist_ok=True)
    config_path = script_dir / "config" / "groups.config.yml"
    config_path.write_text("dummy", encoding="utf-8")

    data_dir = script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "t1.txt").write_text("Vita Iulius æternus.\n", encoding="utf-8")

    out_dir = script_dir / "output"

    cfg = {
        "groups": {"text": {"files": [str(data_dir / "*.txt")]}},
        "out_dir": str(out_dir),
        "analysis_unit": "lemma",
        "nlp": {
            "language": "la",
            "stanza_package": "perseus",
            "cpu_only": True,
        },
        "dictcheck": {"enabled": False},
        "normalization": {
            "enabled": True,
            "casefold": True,
            "map_u_v": True,
            "map_i_j": True,
            "strip_diacritics": True,
            "normalize_ligatures": True,
            "unicode_nf": "NFC",
        },
    }

    def fake_load_config(path: Path):
        assert path.name == "groups.config.yml"
        return cfg


    dependencies = runner_dependencies(
        fake_load_config,
        fake_backend_factory([("x", "x", "NOUN")]),
    )
    # Act
    result = run(
        project_root=script_dir,
        config_path=config_path,
        dependencies=dependencies,
    )
    assert result.exit_code == 0

    # Assert: summary.txt contains policy line
    summary = (out_dir / "summary.txt").read_text(encoding="utf-8")
    assert "normalization:" in summary
    assert "map_u_v=True" in summary
    assert "map_i_j=True" in summary
    assert "strip_diacritics=True" in summary

    # Assert: run_meta.json contains policy dict (structured)
    meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["normalization"]["enabled"] is True
    assert meta["normalization"]["map_u_v"] is True
    assert meta["normalization"]["map_i_j"] is True
    assert meta["normalization"]["strip_diacritics"] is True
    assert meta["normalization"]["normalize_ligatures"] is True
