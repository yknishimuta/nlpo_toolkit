import ast
from pathlib import Path


FEATURES_PATH = Path("nlpo_toolkit/corpus_analysis/features.py")


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _called_names(function: ast.FunctionDef) -> set[str]:
    return {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_features_uses_shared_analysis_record_type() -> None:
    import nlpo_toolkit.corpus_analysis.features as features

    assert not hasattr(features, "TokenRecord")
    assert not hasattr(features, "extract_token_records")


def test_features_has_no_private_token_model_or_document_walker() -> None:
    source = FEATURES_PATH.read_text(encoding="utf-8")

    assert "class TokenRecord" not in source
    assert "def extract_token_records" not in source
    assert "_ROMAN_RE" not in source
    assert "evaluate_analysis_record" not in source


def test_features_imports_only_the_shared_record_extraction_boundary() -> None:
    tree = ast.parse(FEATURES_PATH.read_text(encoding="utf-8"))
    imports = {
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert ("analysis_records", "NLPAnalysisRecord") in imports
    assert ("analysis_records", "iter_nlp_analysis_records_from_text") in imports
    assert ("nlpo_toolkit.nlp.roman_numerals", "should_drop_roman_numeral") in imports
    assert not any(
        module in {"runner", "token_artifact", "diagnostic_trace"}
        for module, _name in imports
    )


def test_feature_calculators_do_not_filter_records() -> None:
    tree = ast.parse(FEATURES_PATH.read_text(encoding="utf-8"))
    forbidden = {
        "filter_feature_records",
        "should_drop_roman_numeral",
        "is_word_token_text",
    }
    for name in (
        "compute_basic_features",
        "compute_upos_features",
        "select_mfw",
        "compute_mfw_features",
    ):
        assert not _called_names(_function(tree, name)) & forbidden


def test_feature_filter_has_one_production_call_site() -> None:
    tree = ast.parse(FEATURES_PATH.read_text(encoding="utf-8"))
    callers = [
        function.name
        for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        if "filter_feature_records" in _called_names(function)
    ]
    assert callers == ["build_feature_rows"]


def test_features_do_not_use_count_selection_or_resolve_reference_paths() -> None:
    source = FEATURES_PATH.read_text(encoding="utf-8")
    assert "filters.upos_targets" not in source
    assert "evaluate_analysis_record" not in source
    assert "resolve_project_path" not in source
    assert 'config.filters.roman_exceptions_file' not in source
