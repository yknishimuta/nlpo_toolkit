from __future__ import annotations

from pathlib import Path

from .support.module_graph import (
    build_module_graph,
    collapse_graph,
    find_cycles,
    find_forbidden_dependencies,
)
from .support.rules import DependencyRule, format_violations
from .support.source_checks import (
    find_cross_module_private_imports,
    find_forbidden_identifiers,
    find_generic_class_bases,
    find_mutable_fields_in_frozen_models,
)
from .support.module_roles import (
    ModuleRole,
    ModuleRolePolicy,
    find_module_role_issues,
    find_stale_role_selectors,
    roles_for_module,
    validate_role_policies,
)


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


def test_mutable_field_check_targets_only_frozen_value_model_fields(tmp_path: Path) -> None:
    root = tmp_path / "sample"
    source = root / "models.py"
    _write(
        source,
        """from dataclasses import dataclass
@dataclass(frozen=True)
class FrozenValue:
    bad: list[str]
    good: tuple[str, ...]
@dataclass
class Builder:
    allowed: list[str]
def build() -> list[str]:
    local: list[str] = []
    return local
""",
    )
    violations = find_mutable_fields_in_frozen_models(
        (source,), package_root=root, package_name="sample"
    )
    assert len(violations) == 1
    assert "class: FrozenValue" in violations[0].qualified_name
    assert "field: bad" in violations[0].qualified_name
    assert "annotation: list[str]" in violations[0].qualified_name


def test_forbidden_identifier_check_finds_definitions_fields_imports_and_accesses(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.py"
    _write(
        source,
        """from package import old_import
__all__ = ["old_function"]
class Model:
    old_field: str
def old_function():
    return module.old_attribute
""",
    )
    violations = find_forbidden_identifiers(
        (source,),
        names={"old_import", "old_field", "old_function", "old_attribute"},
    )
    assert {item.qualified_name for item in violations} == {
        "old_import", "old_field", "old_function", "old_attribute"
    }


def test_generic_class_base_check_is_limited_to_selected_classes(tmp_path: Path) -> None:
    source = tmp_path / "results.py"
    _write(
        source,
        """from typing import Generic, TypeVar
T = TypeVar("T")
class Selected(Generic[T]):
    pass
class Unrelated(Generic[T]):
    pass
""",
    )
    violations = find_generic_class_bases((source,), class_names={"Selected"})
    assert [item.qualified_name for item in violations] == ["Selected"]


def test_module_role_exact_and_recursive_matching_is_segment_aware() -> None:
    policies = (
        ModuleRolePolicy(ModuleRole.DOMAIN, exact_modules=("sample.domain",)),
        ModuleRolePolicy(ModuleRole.INFRASTRUCTURE, recursive_packages=("sample.adapters",)),
    )
    assert roles_for_module("sample.domain", policies) == {ModuleRole.DOMAIN}
    assert roles_for_module("sample.domain_extra", policies) == set()
    for module in ("sample.adapters", "sample.adapters.file", "sample.adapters.nested.writer"):
        assert roles_for_module(module, policies) == {ModuleRole.INFRASTRUCTURE}
    assert roles_for_module("sample.adapters_extra", policies) == set()


def test_module_role_issues_report_missing_overlap_and_stable_order() -> None:
    policies = (
        ModuleRolePolicy(ModuleRole.APPLICATION, exact_modules=("sample.overlap",)),
        ModuleRolePolicy(ModuleRole.INFRASTRUCTURE, exact_modules=("sample.overlap",)),
    )
    issues = find_module_role_issues(("sample.z_missing", "sample.overlap", "sample.a_missing"), policies)
    assert [issue.module for issue in issues] == ["sample.a_missing", "sample.overlap", "sample.z_missing"]
    assert [issue.kind for issue in issues] == ["unclassified-module", "multiply-classified-module", "unclassified-module"]
    assert issues[1].matched_roles == (ModuleRole.APPLICATION, ModuleRole.INFRASTRUCTURE)


def test_stale_role_selectors_detect_exact_and_recursive() -> None:
    policies = (
        ModuleRolePolicy(ModuleRole.DOMAIN, exact_modules=("sample.present", "sample.removed")),
        ModuleRolePolicy(ModuleRole.INFRASTRUCTURE, recursive_packages=("sample.adapters", "sample.old_adapters")),
    )
    stale = find_stale_role_selectors(("sample.present", "sample.adapters.file"), policies)
    assert [(item.selector_kind, item.selector) for item in stale] == [
        ("exact", "sample.removed"),
        ("recursive", "sample.old_adapters"),
    ]


def test_role_policy_validation_rejects_bad_selectors() -> None:
    policies = (
        ModuleRolePolicy(
            ModuleRole.SHARED,
            exact_modules=("", "outside.project", "sample.duplicate", "sample.duplicate"),
            recursive_packages=("nlpo_toolkit", "nlpo_toolkit.bad.*", "nlpo_toolkit/bad"),
        ),
        ModuleRolePolicy(
            ModuleRole.DOMAIN,
            exact_modules=("sample.duplicate",),
        ),
    )
    messages = "\n".join(str(problem) for problem in validate_role_policies(policies))
    assert "duplicate exact selector" in messages
    assert "empty exact selector" in messages
    assert "must start with nlpo_toolkit" in messages
    assert "catch-all" in messages
    assert "invalid recursive selector syntax" in messages
    assert "multiple roles" in messages
