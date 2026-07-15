from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path("nlpo_toolkit/corpus_analysis/features")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }


def test_features_is_a_non_facade_responsibility_package() -> None:
    assert not Path("nlpo_toolkit/corpus_analysis/features.py").exists()
    assert ROOT.is_dir()
    init = (ROOT / "__init__.py").read_text(encoding="utf-8")
    assert "ImportFrom" not in ast.dump(ast.parse(init))


def test_shared_record_extraction_is_owned_only_by_engine() -> None:
    owners = [
        path.name
        for path in ROOT.glob("*.py")
        if "iter_nlp_analysis_records_from_text" in _imports(path)
    ]
    assert owners == ["engine.py"]
    assert "NLPAnalysisRecord" in _imports(ROOT / "filtering.py")


def test_calculation_modules_do_not_depend_on_application_services() -> None:
    forbidden = {
        "FeatureCommandDependencies", "CorpusExecutionSession", "NLPExecutionSession",
        "prepare_analysis_corpus_session", "start_nlp_execution_session",
    }
    for name in ("filtering.py", "lexical.py", "upos.py", "mfw.py", "engine.py"):
        assert _imports(ROOT / name).isdisjoint(forbidden)


def test_service_is_thin_and_calculation_has_no_any_or_old_tuple_input() -> None:
    service = (ROOT / "service.py").read_text(encoding="utf-8")
    assert not any(token in service for token in (
        "Counter", "compute_basic_features", "compute_upos_features", "select_mfw",
        "compute_mfw_features", "filter_feature_records",
        "iter_nlp_analysis_records_from_text", "safe_feature_name",
        "build_analysis_plan", "prepare_analysis_plan", "prepare_corpora",
        "load_roman_exceptions",
    ))
    all_source = "\n".join(path.read_text(encoding="utf-8") for path in ROOT.glob("*.py"))
    assert "groups_texts" not in all_source
    assert "typing import Any" not in all_source
    assert "dict[str, Any]" not in all_source


def test_features_do_not_use_count_selection() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in ROOT.glob("*.py"))
    assert "filters.upos_targets" not in source
    assert "evaluate_analysis_record" not in source
    assert "resolve_project_path" not in source
