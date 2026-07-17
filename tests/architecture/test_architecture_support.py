from __future__ import annotations

from pathlib import Path

from .support.module_graph import (
    build_module_graph,
    collapse_graph,
    find_cycles,
    find_forbidden_dependencies,
)
from .support.rules import DependencyRule, format_violations
from .support.source_checks import find_cross_module_private_imports


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_graph_resolves_all_static_import_forms(tmp_path: Path) -> None:
    root = tmp_path / "sample"
    _write(root / "__init__.py", "from . import models\n")
    _write(root / "models.py", "class Model: pass\n")
    _write(root / "nested" / "__init__.py", "")
    _write(root / "nested" / "worker.py", """
from typing import TYPE_CHECKING
import sample.models as aliased
from ..models import Model
if TYPE_CHECKING:
    from sample.models import Model as CheckedModel
def run():
    from .. import models
""")
    graph = build_module_graph(root, package_name="sample")
    worker_edges = [edge for edge in graph.edges if edge.importer == "sample.nested.worker"]
    assert {edge.imported for edge in worker_edges} == {"sample.models"}
    assert len(worker_edges) == 4
    assert any(edge.importer == "sample" and edge.imported == "sample.models" for edge in graph.edges)
    assert all(not edge.imported.startswith("typing") for edge in graph.edges)


def test_symbol_import_distinguishes_submodule_from_name(tmp_path: Path) -> None:
    root = tmp_path / "sample"
    _write(root / "__init__.py", "")
    _write(root / "models.py", "VALUE = 1\n")
    _write(root / "consumer.py", "from sample import models\nfrom sample.models import VALUE\n")
    graph = build_module_graph(root, package_name="sample")
    edges = [edge for edge in graph.edges if edge.importer == "sample.consumer"]
    assert any(edge.imported == "sample.models" and edge.imported_name is None for edge in edges)
    assert any(edge.imported == "sample.models" and edge.imported_name == "VALUE" for edge in edges)


def test_dynamic_imports_are_recorded_or_rejected(tmp_path: Path) -> None:
    root = tmp_path / "sample"
    _write(root / "__init__.py", "")
    _write(root / "plugin.py", "")
    _write(root / "loader.py", """
import importlib
importlib.import_module("sample.plugin")
__import__("sample.plugin")
def load(name):
    return importlib.import_module(name)
""")
    graph = build_module_graph(root, package_name="sample")
    dynamic = [edge for edge in graph.edges if edge.is_dynamic]
    assert len(dynamic) == 2
    assert {edge.imported for edge in dynamic} == {"sample.plugin"}
    assert len(graph.dynamic_import_issues) == 1
    assert graph.dynamic_import_issues[0].line_number == 6


def test_module_and_collapsed_package_cycles_are_detected(tmp_path: Path) -> None:
    root = tmp_path / "sample"
    _write(root / "__init__.py", "")
    _write(root / "left" / "__init__.py", "")
    _write(root / "left" / "a.py", "from sample.right import b\n")
    _write(root / "right" / "__init__.py", "")
    _write(root / "right" / "b.py", "from sample.left import a\n")
    graph = build_module_graph(root, package_name="sample")
    assert find_cycles(graph) == (("sample.left.a", "sample.right.b"),)
    packages = collapse_graph(graph, ("sample.left", "sample.right"))
    assert find_cycles(packages) == (("sample.left", "sample.right"),)


def test_dependency_allowlist_and_diagnostics(tmp_path: Path) -> None:
    root = tmp_path / "sample"
    _write(root / "__init__.py", "")
    _write(root / "ui.py", "")
    _write(root / "service.py", "from sample import ui\n")
    graph = build_module_graph(root, package_name="sample")
    forbidden = DependencyRule("service-no-ui", ("sample.service",), ("sample.ui",), explanation="Services are UI-independent.")
    violations = find_forbidden_dependencies(graph, (forbidden,))
    message = format_violations(violations)
    assert "service-no-ui" in message
    assert "service.py:1" in message
    assert "Services are UI-independent" in message
    allowed = DependencyRule("service-no-ui", ("sample.service",), ("sample.ui",), ("sample.ui",))
    assert find_forbidden_dependencies(graph, (allowed,)) == ()


def test_private_import_check_is_project_scoped(tmp_path: Path) -> None:
    source = tmp_path / "consumer.py"
    _write(source, "from sample.module import _secret\nimport sample.module as module\nmodule._other()\nfrom os import _exit\n")
    violations = find_cross_module_private_imports((source,), project_prefix="sample")
    assert {item.qualified_name for item in violations} == {"sample.module._secret", "sample.module._other"}

