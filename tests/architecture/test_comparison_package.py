import ast
from pathlib import Path


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_only_one_comparison_package_exists() -> None:
    assert Path("nlpo_toolkit/comparison").is_dir()
    assert not (Path("nlpo_toolkit") / "compare").exists()
    assert not (Path("nlpo_toolkit/corpus_analysis") / "comparison.py").exists()


def test_comparison_package_does_not_depend_on_corpus_analysis() -> None:
    for path in Path("nlpo_toolkit/comparison").glob("*.py"):
        assert not any("corpus_analysis" in module for module in _imports(path)), path


def test_engine_modules_do_not_import_io_or_application_services() -> None:
    forbidden = {"csv", "json", "pathlib", "sys", "argparse", "frequency_io", "cli_service", "configured", "writers"}
    for name in ("models.py", "metrics.py", "engine.py"):
        imports = _imports(Path("nlpo_toolkit/comparison") / name)
        assert not any(module.split(".")[-1] in forbidden for module in imports), name


def test_corpus_analysis_has_no_relative_comparison_import() -> None:
    for path in Path("nlpo_toolkit/corpus_analysis").glob("*.py"):
        assert "from .comparison import" not in path.read_text(encoding="utf-8")
