import ast
from pathlib import Path


ROOT = Path("nlpo_toolkit/corpus_analysis/token_artifact")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }


def test_monolith_is_replaced_by_non_facade_package() -> None:
    assert not Path("nlpo_toolkit/corpus_analysis/token_artifact.py").exists()
    assert {path.name for path in ROOT.glob("*.py")} >= {
        "__init__.py", "errors.py", "schema.py", "paths.py", "codec.py",
        "integrity.py", "writer.py", "reader.py", "validation.py",
    }
    init = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in init.body)


def test_package_dependency_direction() -> None:
    schema = _imports(ROOT / "schema.py")
    codec = _imports(ROOT / "codec.py")
    writer = _imports(ROOT / "writer.py")
    reader = _imports(ROOT / "reader.py")
    assert not any(name.endswith(("writer", "reader", "validation")) for name in schema)
    assert not any(name.endswith(("writer", "reader", "validation")) for name in codec)
    assert not any(name.endswith("reader") for name in writer)
    assert not any(name.endswith("writer") for name in reader)


def test_package_has_no_cli_count_or_artifact_plan_dependency() -> None:
    forbidden = ("cli", "artifacts", "runner", "analysis_execution")
    offenders = []
    for path in ROOT.glob("*.py"):
        for module in _imports(path):
            if any(name in module for name in forbidden):
                offenders.append((str(path), module))
    assert offenders == []


def test_row_and_csv_responsibilities_have_single_owners() -> None:
    definitions = {}
    dict_writers = []
    dict_readers = []
    for path in ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions[path.name] = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "DictWriter":
                dict_writers.append(path.name)
            if isinstance(node, ast.Attribute) and node.attr == "DictReader":
                dict_readers.append(path.name)
    assert "encode_token_record" in definitions["codec.py"]
    assert "decode_token_record" in definitions["codec.py"]
    assert set(dict_writers) == {"writer.py"}
    assert set(dict_readers) == {"reader.py"}


def test_production_has_no_old_module_import_shape_or_metadata_coercion() -> None:
    offenders = []
    for path in Path("nlpo_toolkit").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("token_artifact"):
                offenders.append(str(path))
    assert offenders == []
    source = "\n".join(path.read_text(encoding="utf-8") for path in ROOT.glob("*.py"))
    assert "bool(data.get" not in source
    assert "int(data.get" not in source
    assert "str(data.get" not in source
