# tests/test_run_clean_corpus.py

from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlpo_toolkit.latin.cleaners import run_clean_corpus as mod


def test_main_uses_default_config(tmp_path, monkeypatch):
    """
    When no argv is passed, main() should:
      - use DEFAULT_CONFIG
      - resolve input/output paths relative to the config file directory
      - read the input text
      - pass it (and kind, ref_tsv, doc_id, rules_path) to clean_text
      - write the cleaned text to the resolved output path
    """

    # Prepare a fake config file and input/output locations
    config_dir = tmp_path / "cfg"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / "sample.yml"
    config_path.write_text("dummy: true\n", encoding="utf-8")

    input_path = config_dir / "input.txt"
    input_path.write_text("Salve mundi", encoding="utf-8")

    expected_output = (config_dir / "out" / "cleaned.txt").resolve()

    monkeypatch.setattr(mod, "DEFAULT_CONFIG", config_path)

    # (NEW) prepare externalized rules YAML
    rules_rel = Path("config/latin_cleaners/corpus_corporum.yml")
    rules_abs = (config_dir / rules_rel).resolve()
    rules_abs.parent.mkdir(parents=True, exist_ok=True)
    rules_abs.write_text("# dummy rules\n", encoding="utf-8")

    # Monkeypatch load_clean_config to return a controlled mapping
    def fake_load_clean_config(path: Path):
        # Ensure that the script passes the expected config path
        assert path == config_path
        return {
            "kind": "corpus_corporum",
            "input": "input.txt",
            "output": "out/cleaned.txt",
            "rules_path": str(rules_rel),
            # ref_tsv/doc_id_prefix intentionally omitted in this test
        }

    monkeypatch.setattr(mod, "load_clean_config", fake_load_clean_config)

    def fake_clean_text(
        raw: str,
        *,
        kind: str,
        ref_tsv=None,
        doc_id: str = "",
        rules_path=None,
    ) -> str:
        # Verify that raw and kind are passed correctly
        assert raw == "Salve mundi"
        assert kind == "corpus_corporum"

        # assertions
        assert ref_tsv is None
        assert doc_id == "input"  # stem of input.txt

        # (NEW) rules_path should be resolved relative to config_dir
        assert Path(rules_path) == rules_abs

        return raw.upper()

    monkeypatch.setattr(mod, "clean_text", fake_clean_text)

    rc = mod.main(argv=[])
    assert rc == 0

    # Verify the output
    assert expected_output.is_file(), f"Expected output file not found: {expected_output}"
    out_text = expected_output.read_text(encoding="utf-8")
    assert out_text == "SALVE MUNDI"


def test_main_with_explicit_config_path(tmp_path, monkeypatch):
    """
    When argv contains a path, main() should:
      - use that path instead of DEFAULT_CONFIG
      - still resolve input/output relative to the config file directory
      - pass kind/ref_tsv/doc_id/rules_path to clean_text
    """

    # Prepare a separate fake config and input/output under tmp_path
    config_dir = tmp_path / "cfg2"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / "custom.yml"
    config_path.write_text("dummy: true\n", encoding="utf-8")

    input_path = config_dir / "in.txt"
    input_path.write_text("Puella rosam amat.", encoding="utf-8")

    expected_output = (config_dir / "out" / "cleaned2.txt").resolve()

    # (NEW) prepare externalized rules YAML
    rules_rel = Path("config/latin_cleaners/custom_rules.yml")
    rules_abs = (config_dir / rules_rel).resolve()
    rules_abs.parent.mkdir(parents=True, exist_ok=True)
    rules_abs.write_text("# dummy rules\n", encoding="utf-8")

    def fake_load_clean_config(path: Path):
        # main() should pass the argv[0] here
        assert path == config_path
        return {
            "kind": "sample_kind",
            "input": "in.txt",
            "output": "out/cleaned2.txt",
            "rules_path": str(rules_rel),
            # ref_tsv/doc_id_prefix intentionally omitted in this test
        }

    monkeypatch.setattr(mod, "load_clean_config", fake_load_clean_config)

    def fake_clean_text(
        raw: str,
        *,
        kind: str,
        ref_tsv=None,
        doc_id: str = "",
        rules_path=None,
    ) -> str:
        assert raw == "Puella rosam amat."
        assert kind == "sample_kind"

        # assertions
        assert ref_tsv is None
        assert doc_id == "in"  # stem of in.txt

        # (NEW) rules_path should be resolved relative to config_dir
        assert Path(rules_path) == rules_abs

        return raw.replace(" ", "_")

    monkeypatch.setattr(mod, "clean_text", fake_clean_text)

    rc = mod.main(argv=[str(config_path)])
    assert rc == 0

    # Verify output
    assert expected_output.is_file()
    out_text = expected_output.read_text(encoding="utf-8")
    assert out_text == "Puella_rosam_amat."