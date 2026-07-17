from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .rules import ArchitectureViolation, DependencyRule, matches_prefix


@dataclass(frozen=True)
class ImportEdge:
    importer: str
    imported: str
    source_path: Path
    line_number: int
    imported_name: str | None = None
    is_dynamic: bool = False


@dataclass(frozen=True)
class DynamicImportIssue:
    importer: str
    source_path: Path
    line_number: int
    call_name: str

    def __str__(self) -> str:
        return (
            "[non-literal-dynamic-import]\n"
            f"{self.importer} calls {self.call_name} with a non-literal target\n"
            f"at {self.source_path}:{self.line_number}"
        )


@dataclass(frozen=True)
class ModuleGraph:
    modules: frozenset[str]
    edges: tuple[ImportEdge, ...]
    dynamic_import_issues: tuple[DynamicImportIssue, ...] = ()


def _module_name(path: Path, root: Path, package_name: str) -> str:
    relative = path.relative_to(root)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join((package_name, *parts)) if parts else package_name


def _resolve_relative(
    importer: str, *, level: int, module: str | None, is_package: bool = False
) -> str:
    package = importer if is_package else importer.rpartition(".")[0]
    parts = package.split(".") if package else []
    if level > 1:
        parts = parts[: -(level - 1)]
    if module:
        parts.extend(module.split("."))
    return ".".join(parts)


def _dynamic_call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name) and node.func.id == "__import__":
        return "__import__"
    if (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "importlib"
        and node.func.attr == "import_module"
    ):
        return "importlib.import_module"
    return None


def build_module_graph(package_root: Path, *, package_name: str) -> ModuleGraph:
    paths = tuple(sorted(package_root.rglob("*.py")))
    module_by_path = {path: _module_name(path, package_root, package_name) for path in paths}
    modules = frozenset(module_by_path.values())
    edges: list[ImportEdge] = []
    issues: list[DynamicImportIssue] = []
    for path, importer in module_by_path.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if matches_prefix(alias.name, package_name):
                        edges.append(ImportEdge(importer, alias.name, path, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                base = (
                    _resolve_relative(
                        importer,
                        level=node.level,
                        module=node.module,
                        is_package=path.name == "__init__.py",
                    )
                    if node.level
                    else (node.module or "")
                )
                if not matches_prefix(base, package_name):
                    continue
                for alias in node.names:
                    candidate = f"{base}.{alias.name}" if base else alias.name
                    if candidate in modules:
                        edges.append(ImportEdge(importer, candidate, path, node.lineno))
                    else:
                        edges.append(
                            ImportEdge(importer, base, path, node.lineno, alias.name)
                        )
            elif isinstance(node, ast.Call):
                call_name = _dynamic_call_name(node)
                if call_name is None:
                    continue
                target = node.args[0] if node.args else None
                if isinstance(target, ast.Constant) and isinstance(target.value, str):
                    imported = target.value
                    if imported.startswith(".") and call_name == "importlib.import_module":
                        imported = _resolve_relative(
                            importer,
                            level=len(imported) - len(imported.lstrip(".")),
                            module=imported.lstrip("."),
                            is_package=path.name == "__init__.py",
                        )
                    if matches_prefix(imported, package_name):
                        edges.append(
                            ImportEdge(
                                importer, imported, path, node.lineno, is_dynamic=True
                            )
                        )
                else:
                    issues.append(DynamicImportIssue(importer, path, node.lineno, call_name))
    return ModuleGraph(
        modules,
        tuple(sorted(edges, key=lambda edge: (edge.importer, edge.imported, edge.line_number))),
        tuple(sorted(issues, key=lambda issue: (issue.importer, issue.line_number))),
    )


def dependencies_of(graph: ModuleGraph, module_or_prefix: str) -> frozenset[str]:
    return frozenset(
        edge.imported
        for edge in graph.edges
        if matches_prefix(edge.importer, module_or_prefix)
    )


def find_forbidden_dependencies(
    graph: ModuleGraph, rules: Sequence[DependencyRule]
) -> tuple[ArchitectureViolation, ...]:
    violations: list[ArchitectureViolation] = []
    for rule in rules:
        for edge in graph.edges:
            if not any(matches_prefix(edge.importer, item) for item in rule.source_prefixes):
                continue
            if any(matches_prefix(edge.importer, item) for item in rule.excluded_source_prefixes):
                continue
            if any(matches_prefix(edge.imported, item) for item in rule.allowed_target_prefixes):
                continue
            if any(
                matches_prefix(edge.imported, item)
                for item in rule.forbidden_target_prefixes
            ):
                violations.append(
                    ArchitectureViolation(
                        rule.name,
                        edge.importer,
                        edge.imported,
                        edge.source_path,
                        edge.line_number,
                        rule.explanation,
                    )
                )
    return tuple(sorted(set(violations), key=lambda item: (item.rule_name, item.importer, item.line_number)))


def _adjacency(graph: ModuleGraph) -> dict[str, set[str]]:
    result = {module: set() for module in graph.modules}
    for edge in graph.edges:
        if edge.imported in graph.modules and edge.imported != edge.importer:
            result[edge.importer].add(edge.imported)
    return result


def find_cycles(graph: ModuleGraph) -> tuple[tuple[str, ...], ...]:
    adjacency = _adjacency(graph)
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(module: str) -> None:
        nonlocal index
        indices[module] = lowlinks[module] = index
        index += 1
        stack.append(module)
        on_stack.add(module)
        for target in sorted(adjacency[module]):
            if target not in indices:
                visit(target)
                lowlinks[module] = min(lowlinks[module], lowlinks[target])
            elif target in on_stack:
                lowlinks[module] = min(lowlinks[module], indices[target])
        if lowlinks[module] == indices[module]:
            component: list[str] = []
            while True:
                item = stack.pop()
                on_stack.remove(item)
                component.append(item)
                if item == module:
                    break
            if len(component) > 1:
                components.append(tuple(sorted(component)))

    for module in sorted(graph.modules):
        if module not in indices:
            visit(module)
    return tuple(sorted(components))


def collapse_graph(graph: ModuleGraph, package_prefixes: Iterable[str]) -> ModuleGraph:
    prefixes = tuple(sorted(package_prefixes, key=len, reverse=True))

    def owner(module: str) -> str | None:
        return next((prefix for prefix in prefixes if matches_prefix(module, prefix)), None)

    edges: list[ImportEdge] = []
    for edge in graph.edges:
        source, target = owner(edge.importer), owner(edge.imported)
        if source and target and source != target:
            edges.append(
                ImportEdge(source, target, edge.source_path, edge.line_number, edge.imported_name, edge.is_dynamic)
            )
    return ModuleGraph(frozenset(prefixes), tuple(edges))


def cycle_diagnostics(graph: ModuleGraph, cycles: Iterable[tuple[str, ...]]) -> str:
    blocks: list[str] = []
    for component in cycles:
        members = set(component)
        relevant = [
            edge for edge in graph.edges
            if edge.importer in members and edge.imported in members
        ]
        lines = ["Import cycle:", *[f"  {module}" for module in component], "Edges:"]
        lines.extend(
            f"  {edge.importer} -> {edge.imported} at {edge.source_path}:{edge.line_number}"
            for edge in relevant
        )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
