from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Collection, Iterable


@dataclass(frozen=True)
class SourceViolation:
    rule_name: str
    source_path: Path
    line_number: int
    qualified_name: str
    explanation: str = ""

    def __str__(self) -> str:
        detail = (
            f"[{self.rule_name}]\n{self.qualified_name}\n"
            f"at {self.source_path}:{self.line_number}"
        )
        return f"{detail}\n\n{self.explanation}" if self.explanation else detail


@lru_cache(maxsize=None)
def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _trees(paths: Iterable[Path]):
    for path in sorted(paths):
        yield path, _tree(path)


def _aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".")[0]] = item.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    return aliases


def _qualified(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        parent = _qualified(node.value, aliases)
        return f"{parent}.{node.attr}" if parent else None
    return None


def find_calls(
    paths: Iterable[Path], *, qualified_names: Collection[str], rule_name: str = "forbidden-call"
) -> tuple[SourceViolation, ...]:
    violations: list[SourceViolation] = []
    for path, tree in _trees(paths):
        aliases = _aliases(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _qualified(node.func, aliases)
                if name in qualified_names:
                    violations.append(SourceViolation(rule_name, path, node.lineno, name))
    return tuple(sorted(violations, key=lambda item: (str(item.source_path), item.line_number)))


def find_imports(
    paths: Iterable[Path], *, module_prefixes: Collection[str], rule_name: str = "forbidden-import"
) -> tuple[SourceViolation, ...]:
    violations: list[SourceViolation] = []
    for path, tree in _trees(paths):
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if any(name == prefix or name.startswith(prefix + ".") for prefix in module_prefixes):
                    violations.append(SourceViolation(rule_name, path, node.lineno, name))
    return tuple(sorted(violations, key=lambda item: (str(item.source_path), item.line_number)))


def find_attribute_accesses(
    paths: Iterable[Path], *, qualified_names: Collection[str], rule_name: str = "forbidden-access"
) -> tuple[SourceViolation, ...]:
    violations: list[SourceViolation] = []
    for path, tree in _trees(paths):
        aliases = _aliases(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                name = _qualified(node, aliases)
                if name in qualified_names:
                    violations.append(SourceViolation(rule_name, path, node.lineno, name))
    return tuple(sorted(violations, key=lambda item: (str(item.source_path), item.line_number)))


def find_cross_module_private_imports(
    paths: Iterable[Path], *, project_prefix: str = "nlpo_toolkit"
) -> tuple[SourceViolation, ...]:
    violations: list[SourceViolation] = []
    for path, tree in _trees(paths):
        module_aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                is_project_import = node.level > 0 or bool(
                    node.module
                    and (
                        node.module == project_prefix
                        or node.module.startswith(project_prefix + ".")
                    )
                )
                if not is_project_import:
                    continue
                for item in node.names:
                    if item.name.startswith("_") and item.name != "__all__":
                        module = node.module or "." * node.level
                        violations.append(SourceViolation("cross-module-private-import", path, node.lineno, f"{module}.{item.name}"))
            elif isinstance(node, ast.Import):
                for item in node.names:
                    if item.name == project_prefix or item.name.startswith(project_prefix + "."):
                        module_aliases[item.asname or item.name.split(".")[-1]] = item.name
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr.startswith("_") and isinstance(node.value, ast.Name) and node.value.id in module_aliases:
                violations.append(SourceViolation("cross-module-private-access", path, node.lineno, f"{module_aliases[node.value.id]}.{node.attr}"))
    return tuple(sorted(violations, key=lambda item: (str(item.source_path), item.line_number, item.qualified_name)))
