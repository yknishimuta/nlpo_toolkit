import json
from pathlib import Path

from nlpo_toolkit.corpus_analysis import cli
from nlpo_toolkit.corpus_analysis.cli import count as mod
from tests.corpus_analysis.fake_nlp import fake_backend_factory


def test_normalization_policy_is_written_to_summary_and_run_meta(tmp_path, monkeypatch):
    # Arrange
    script_dir = tmp_path / "runner_dir"
    (script_dir / "config").mkdir(parents=True, exist_ok=True)

    data_dir = script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "t1.txt").write_text("Vita Iulius æternus.\n", encoding="utf-8")

    out_dir = script_dir / "output"

    cfg = {
        "groups": {"text": {"files": [str(data_dir / "*.txt")]}},
        "out_dir": str(out_dir),
        "analysis_unit": "lemma",
        "language": "la",
        "stanza_package": "perseus",
        "cpu_only": True,
        "dictcheck": {"enabled": False},
        "normalization": {
            "enabled": True,
            "casefold": True,
            "uv": "v_to_u",
            "ij": "j_to_i",
            "diacritics": "strip",
            "ligatures": {"æ": "ae", "œ": "oe"},
        },
    }

    def fake_load_config(path: Path):
        assert path.name == "groups.config.yml"
        return cfg

    monkeypatch.setattr(mod, "load_config", fake_load_config)

    # Make main() think config exists
    real_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if self.name == "groups.config.yml":
            return True
        return real_exists(self)

    monkeypatch.setattr(mod.Path, "exists", fake_exists)

    
    monkeypatch.setattr(mod, "create_nlp_backend", fake_backend_factory([("x", "x", "NOUN")]))
    monkeypatch.setattr(mod, "render_stanza_package_table", lambda *a, **k: ["[stanza stub]"])

    # Act
    rc = cli.main(["count-vocabula", "--project-root", str(script_dir)])
    assert rc == 0

    # Assert: summary.txt contains policy line
    summary = (out_dir / "summary.txt").read_text(encoding="utf-8")
    assert "normalization:" in summary
    assert "uv=v_to_u" in summary
    assert "ij=j_to_i" in summary
    assert "diacritics=strip" in summary

    # Assert: run_meta.json contains policy dict (structured)
    meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["normalization"]["enabled"] is True
    assert meta["normalization"]["uv"] == "v_to_u"
    assert meta["normalization"]["ij"] == "j_to_i"
    assert meta["normalization"]["diacritics"] == "strip"
    assert meta["normalization"]["ligatures"]["æ"] == "ae"
