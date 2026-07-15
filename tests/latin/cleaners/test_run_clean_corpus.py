from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import nlpo_toolkit.latin.cleaners.run_clean_corpus as mod
from nlpo_toolkit.latin.cleaners.models import (
    CleanerProfile,
    CleanerProgram,
    CleaningResult,
    RuleSet,
)


def _program(kind="corpus_corporum") -> CleanerProgram:
    profile = CleanerProfile(kind, Path("rules.yml"), lambda text: tuple(text.splitlines()), lambda line: line)
    return CleanerProgram(profile, RuleSet(), MappingProxyType({}))


def test_main_uses_default_config_and_loads_program_once(tmp_path, monkeypatch):
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    config_path = config_dir / "sample.yml"
    config_path.write_text(
        "kind: corpus_corporum\ninput: input.txt\noutput: out/cleaned.txt\nrules_path: rules.yml\n",
        encoding="utf-8",
    )
    (config_dir / "input.txt").write_text("Salve mundi", encoding="utf-8")
    (config_dir / "rules.yml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(mod, "DEFAULT_CONFIG", config_path)
    loads = []
    program = _program()
    monkeypatch.setattr(mod, "load_cleaner_program", lambda **kwargs: loads.append(kwargs) or program)
    monkeypatch.setattr(mod, "clean_document", lambda raw, **kwargs: CleaningResult(raw.upper(), ()))

    assert mod.main(argv=[]) == 0
    assert len(loads) == 1
    assert (config_dir / "out/cleaned.txt").read_text(encoding="utf-8") == "SALVE MUNDI"


def test_directory_mode_reuses_program_and_passes_doc_ids(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.mkdir()
    (source / "a.txt").write_text("a", encoding="utf-8")
    (source / "b.txt").write_text("b", encoding="utf-8")
    config = tmp_path / "config.yml"
    config.write_text(
        "kind: scholastic_text\ninput: input\noutput: output\ndoc_id_prefix: DOC\n",
        encoding="utf-8",
    )
    program = _program("scholastic_text")
    loads = []
    docs = []
    monkeypatch.setattr(mod, "load_cleaner_program", lambda **kwargs: loads.append(kwargs) or program)

    def clean(raw, **kwargs):
        docs.append(kwargs["doc_id"])
        return CleaningResult(raw.upper(), ())

    monkeypatch.setattr(mod, "clean_document", clean)
    assert mod.main([str(config)]) == 0
    assert len(loads) == 1
    assert docs == ["DOC:a", "DOC:b"]
    assert (tmp_path / "output/a.txt").read_text(encoding="utf-8") == "A"
    assert (tmp_path / "output/b.txt").read_text(encoding="utf-8") == "B"
