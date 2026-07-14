from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "nlpo_toolkit/nlp"
PRODUCTION = ROOT / "nlpo_toolkit"
UTILITY_NAMES = {
    "iter_char_chunks",
    "normalize_token",
    "load_wordlist",
    "RomanExceptionsError",
    "load_roman_exceptions",
    "normalize_roman_exceptions",
    "effective_roman_exceptions",
    "should_drop_roman_numeral",
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_nlp_is_a_responsibility_split_package() -> None:
    assert not (PRODUCTION / "nlp.py").exists()
    assert PACKAGE.is_dir()
    for name in ("chunking.py", "normalization.py", "roman_numerals.py", "vocabulary.py"):
        assert (PACKAGE / name).is_file()


def test_package_init_is_not_a_compatibility_facade() -> None:
    tree = _tree(PACKAGE / "__init__.py")
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body)
    import nlpo_toolkit.nlp as nlp

    assert not any(hasattr(nlp, name) for name in UTILITY_NAMES)
    assert not hasattr(nlp, "__getattr__")


def test_production_consumers_import_owner_submodules() -> None:
    offenders: list[str] = []
    for path in PRODUCTION.rglob("*.py"):
        tree = _tree(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "nlpo_toolkit.nlp":
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "nlpo_toolkit.nlp":
                        offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert offenders == []


def test_low_level_utilities_have_no_application_or_backend_dependencies() -> None:
    forbidden_roots = {"corpus_analysis", "stanza", "transformers", "backends"}
    offenders: list[str] = []
    for path in PACKAGE.glob("*.py"):
        tree = _tree(path)
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if any(part in forbidden_roots for part in alias.name.split(".")):
                        offenders.append(f"{path.name}:{node.lineno}")
            if module and any(part in forbidden_roots for part in module.split(".")):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


def test_pure_utility_boundaries() -> None:
    chunking = _tree(PACKAGE / "chunking.py")
    normalization = _tree(PACKAGE / "normalization.py")
    vocabulary = _tree(PACKAGE / "vocabulary.py")
    assert not any(isinstance(node, ast.Attribute) and node.attr in {"read_text", "open"} for node in ast.walk(chunking))
    assert not any(isinstance(node, ast.Attribute) and node.attr in {"read_text", "open"} for node in ast.walk(normalization))
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "normalize_token" for node in ast.walk(vocabulary))


def test_removed_symbols_and_backend_wrappers_are_absent() -> None:
    forbidden = {"build_stanza_pipeline", "build_sentence_splitter", "PackageType", "_STRIP_PUNCT"}
    offenders: list[str] = []
    for path in PRODUCTION.rglob("*.py"):
        tree = _tree(path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in forbidden:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            if isinstance(node, (ast.Name, ast.Attribute)):
                name = node.id if isinstance(node, ast.Name) else node.attr
                if name in forbidden:
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert offenders == []


def test_fresh_imports_do_not_load_optional_stanza_package() -> None:
    code = """
import sys
import nlpo_toolkit.nlp.chunking
import nlpo_toolkit.nlp.normalization
import nlpo_toolkit.nlp.roman_numerals
import nlpo_toolkit.nlp.vocabulary
import nlpo_toolkit.backends
import nlpo_toolkit.corpus_analysis.analysis_policy
import nlpo_toolkit.corpus_analysis.analysis_records
import nlpo_toolkit.corpus_analysis.dictcheck
assert "stanza" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
