from __future__ import annotations

from pathlib import Path


def test_obsolete_nlp_adapter_modules_are_removed() -> None:
    module_name = "nlp" + "_adapters.py"
    old_package = "count" + "_vocabula"

    assert not Path("nlpo_toolkit/corpus_analysis", module_name).exists()
    assert not Path("nlpo_toolkit", old_package, module_name).exists()


def test_no_transformers_latin_adapter_references_remain() -> None:
    legacy_name = "Transformers" + "LatinAdapter"
    offenders: list[Path] = []

    for root in (Path("nlpo_toolkit"), Path("tests")):
        for path in root.rglob("*.py"):
            if path == Path(__file__):
                continue
            if legacy_name in path.read_text(encoding="utf-8"):
                offenders.append(path)

    assert offenders == []


def test_no_obsolete_nlp_adapter_references_remain() -> None:
    legacy_name = "nlp" + "_adapters"
    offenders: list[Path] = []

    for root in (Path("nlpo_toolkit"), Path("tests")):
        for path in root.rglob("*.py"):
            if path == Path(__file__):
                continue
            if legacy_name in path.read_text(encoding="utf-8"):
                offenders.append(path)

    assert offenders == []
