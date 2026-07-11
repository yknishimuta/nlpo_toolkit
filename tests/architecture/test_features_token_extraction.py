import ast
from pathlib import Path


FEATURES_PATH = Path("nlpo_toolkit/corpus_analysis/features.py")


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
    assert ("nlpo_toolkit.nlp", "should_drop_roman_numeral") in imports
    assert not any(
        module in {"runner", "analysis_pipeline", "token_artifact", "diagnostic_trace"}
        for module, _name in imports
    )
