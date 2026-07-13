from __future__ import annotations

import ast
from pathlib import Path

from pydantic import BaseModel

from nlpo_toolkit.comparison.configured import ComparisonSpec
from nlpo_toolkit.corpus_analysis.config import AppConfig
from nlpo_toolkit.corpus_analysis.partition_models import PartitionSpec


CONFIG_DIR = Path("nlpo_toolkit/corpus_analysis/config")


def test_removed_handwritten_schema_and_value_modules_stay_deleted() -> None:
    assert not (CONFIG_DIR / "schema.py").exists()
    assert not (CONFIG_DIR / "values.py").exists()

    forbidden = ("KNOWN_", "reject_unknown_keys", "_parse_group_config")
    for path in Path("nlpo_toolkit").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in source, f"{name} found in {path}"


def test_parser_uses_app_config_model_validate_as_canonical_boundary() -> None:
    tree = ast.parse((CONFIG_DIR / "parser.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "AppConfig"
        and call.func.attr == "model_validate"
        for call in calls
    )
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("_parse_")
        for node in ast.walk(tree)
    )


def test_serializer_is_a_thin_model_dump_delegate() -> None:
    source = (CONFIG_DIR / "serializer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert len([node for node in tree.body if isinstance(node, ast.FunctionDef)]) == 1
    assert "to_external_dict" in source
    assert '"groups"' not in source


def test_schema_and_directly_configured_specs_come_from_pydantic() -> None:
    schema = AppConfig.model_json_schema(by_alias=True)
    assert issubclass(ComparisonSpec, BaseModel)
    assert issubclass(PartitionSpec, BaseModel)
    assert schema["additionalProperties"] is False
    assert {"groups", "nlp", "filters", "comparisons", "validations", "analysis_cache"} <= set(
        schema["properties"]
    )
