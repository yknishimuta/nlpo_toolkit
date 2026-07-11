from __future__ import annotations

import ast
from pathlib import Path


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_config_is_a_package() -> None:
    root = Path("nlpo_toolkit/corpus_analysis/config")
    assert root.is_dir()
    assert {"__init__.py", "models.py", "schema.py", "values.py", "parser.py", "serializer.py"} <= {
        path.name for path in root.iterdir()
    }
    assert not Path("nlpo_toolkit/corpus_analysis/config.py").exists()


def test_config_models_and_serializer_are_io_free() -> None:
    root = Path("nlpo_toolkit/corpus_analysis/config")
    assert "yaml" not in _imports(root / "models.py")
    assert "pathlib" not in _imports(root / "models.py")
    assert "yaml" not in _imports(root / "serializer.py")


def test_domain_modules_do_not_parse_raw_config() -> None:
    comparison = Path("nlpo_toolkit/comparison/configured.py").read_text(encoding="utf-8")
    partition = Path("nlpo_toolkit/corpus_analysis/partition_validation.py").read_text(encoding="utf-8")
    assert "parse_comparison_specs" not in comparison
    assert "parse_partition_specs" not in partition
    assert 'config.get("comparisons")' not in comparison
    assert 'config.get("validations")' not in partition
