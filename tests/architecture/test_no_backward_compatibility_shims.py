from __future__ import annotations

import ast
import inspect
from pathlib import Path

from nlpo_toolkit.corpus_analysis import diagnostic_trace, token_artifact
from nlpo_toolkit.corpus_analysis.normalizer import normalize_text
from nlpo_toolkit.corpus_analysis.ports import CountRunner
from nlpo_toolkit.corpus_analysis.planning.build import (
    build_analysis_plan,
    build_count_plan,
)
from nlpo_toolkit.corpus_analysis.runner import run
from nlpo_toolkit.corpus_analysis.runtime import prepare_run_context


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "nlpo_toolkit"
ANALYSIS_RECORD_NAMES = {
    "AnalysisOptions",
    "NLPAnalysisRecord",
    "TokenRecord",
    "counter_from_token_records",
    "evaluate_analysis_record",
    "iter_nlp_analysis_records_from_text",
    "iter_token_records",
}


def _tree(relative_path: str) -> ast.Module:
    path = PROJECT_ROOT / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_level_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_production_callables_do_not_accept_script_dir() -> None:
    for function in (
        build_analysis_plan,
        build_count_plan,
        prepare_run_context,
        run,
        CountRunner.__call__,
    ):
        assert "script_dir" not in inspect.signature(function).parameters

    assert not Path("nlpo_toolkit/corpus_analysis/run_plan.py").exists()


def test_production_code_has_no_script_dir_fallback() -> None:
    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.arg) and node.arg == "script_dir":
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")
            if isinstance(node, ast.Name) and node.id == "script_dir":
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")
    assert offenders == []


def test_compare_strips_only_the_canonical_frequency_prefix() -> None:
    from nlpo_toolkit.comparison.frequency_io import labels_from_paths

    assert labels_from_paths(
        [Path("frequency_text.csv"), Path("noun_frequency_text.csv")]
    ) == ["text", "noun_frequency_text"]

    source = (PACKAGE_ROOT / "comparison/frequency_io.py").read_text(encoding="utf-8")
    assert "noun_frequency_" not in source


def test_cleaner_calls_canonical_clean_document_signature_directly() -> None:
    tree = _tree("nlpo_toolkit/latin/cleaners/service.py")
    function = _top_level_function(tree, "_execute_file")
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "clean_document"
    ]
    assert len(calls) == 1
    assert {keyword.arg for keyword in calls[0].keywords} == {
        "profile",
        "rules",
        "lexicon_map",
        "doc_id",
        "snippet_chars",
    }
    assert all(keyword.arg is not None for keyword in calls[0].keywords)
    assert not any(
        isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Name)
        and node.type.id == "TypeError"
        for node in ast.walk(function)
    )
    assert not any(
        isinstance(node, ast.Name) and node.id == "kwargs"
        for node in ast.walk(function)
    )


def test_token_artifact_exports_only_artifact_api() -> None:
    assert ANALYSIS_RECORD_NAMES.isdisjoint(token_artifact.__all__)
    for name in ANALYSIS_RECORD_NAMES:
        assert not hasattr(token_artifact, name)


def test_consumers_do_not_import_analysis_records_from_token_artifact() -> None:
    offenders: list[tuple[str, str]] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if node.module.endswith("token_artifact"):
                for alias in node.names:
                    if alias.name in ANALYSIS_RECORD_NAMES:
                        offenders.append((str(path.relative_to(PROJECT_ROOT)), alias.name))
    assert offenders == []


def test_normalizer_accepts_only_typed_normalization_config() -> None:
    signature = inspect.signature(normalize_text)
    assert tuple(signature.parameters) == ("text", "config")
    assert signature.parameters["config"].annotation == "NormalizationConfig"

    tree = _tree("nlpo_toolkit/corpus_analysis/normalizer.py")
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "_normalization_value"
        for node in tree.body
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "hasattr"
        for node in ast.walk(tree)
    )
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "Mapping" not in imported_names

    corpus_tree = _tree("nlpo_toolkit/corpus_analysis/corpus.py")
    normalize_calls = [
        node
        for node in ast.walk(corpus_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "normalize_text"
    ]
    assert len(normalize_calls) == 1
    config_argument = normalize_calls[0].args[1]
    assert isinstance(config_argument, ast.Attribute)
    assert config_argument.attr == "normalization"


def test_diagnostic_trace_uses_current_constant_name_only() -> None:
    expected = (
        "label",
        "chunk",
        "sent_idx",
        "token_idx",
        "token_char_start_in_chunk",
        "token_char_start_in_text",
        "sentence",
        "token",
        "lemma",
        "upos",
        "ref_tag",
        "global_row",
    )
    assert diagnostic_trace.DIAGNOSTIC_TRACE_COLUMNS == expected
    assert "DIAGNOSTIC_TRACE_COLUMNS" in diagnostic_trace.__all__
    assert not hasattr(diagnostic_trace, "LEGACY_TRACE_COLUMNS")
    assert "LEGACY_TRACE_COLUMNS" not in diagnostic_trace.__all__


def test_no_compatibility_modules_or_module_getattr_exist() -> None:
    forbidden_names = {"compat.py", "legacy.py", "deprecated.py"}
    assert not any(path.name in forbidden_names for path in PACKAGE_ROOT.rglob("*.py"))

    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "__getattr__"
            for node in tree.body
        ):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []


def test_public_packages_do_not_export_removed_symbols() -> None:
    import nlpo_toolkit.comparison as comparison
    import nlpo_toolkit.corpus_analysis as corpus_analysis
    import nlpo_toolkit.corpus_analysis.config as config
    import nlpo_toolkit.latin.cleaners as cleaners

    removed = ANALYSIS_RECORD_NAMES | {"LEGACY_TRACE_COLUMNS"}
    for module in (corpus_analysis, config, comparison, cleaners, token_artifact):
        exported = set(getattr(module, "__all__", ()))
        assert exported.isdisjoint(removed)
        assert not any(hasattr(module, name) for name in removed)
