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


def _is_frozen_dataclass(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        name = _qualified(decorator.func, {})
        if name not in {"dataclass", "dataclasses.dataclass"}:
            continue
        return any(
            keyword.arg == "frozen"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in decorator.keywords
        )
    return False


def _is_frozen_config_model(node: ast.ClassDef) -> bool:
    if any(_qualified(base, {}) in {"ConfigModel"} for base in node.bases):
        return True
    for statement in node.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "model_config" for target in statement.targets):
            continue
        if isinstance(statement.value, ast.Call) and _qualified(statement.value.func, {}) == "ConfigDict":
            return any(
                keyword.arg == "frozen"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in statement.value.keywords
            )
    return False


def _mutable_annotation_name(annotation: ast.expr) -> str | None:
    for node in ast.walk(annotation):
        if not isinstance(node, ast.Subscript):
            continue
        name = _qualified(node.value, {})
        if name in {"list", "dict", "set", "Counter", "collections.Counter"}:
            return ast.unparse(annotation)
    return None


def find_mutable_fields_in_frozen_models(
    paths: Iterable[Path], *, package_root: Path, package_name: str
) -> tuple[SourceViolation, ...]:
    violations: list[SourceViolation] = []
    for path, tree in _trees(paths):
        relative = path.resolve().relative_to(package_root.resolve())
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        module = ".".join((package_name, *parts))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not (_is_frozen_dataclass(node) or _is_frozen_config_model(node)):
                continue
            for statement in node.body:
                if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
                    continue
                annotation = _mutable_annotation_name(statement.annotation)
                if annotation is None:
                    continue
                violations.append(
                    SourceViolation(
                        "mutable-field-in-frozen-model",
                        path,
                        statement.lineno,
                        f"module: {module}\nclass: {node.name}\nfield: {statement.target.id}\nannotation: {annotation}",
                        "Frozen value models must use immutable collection field types.",
                    )
                )
    return tuple(sorted(violations, key=lambda item: (item.qualified_name, item.line_number)))


def find_forbidden_identifiers(
    paths: Iterable[Path], *, names: Collection[str]
) -> tuple[SourceViolation, ...]:
    violations: list[SourceViolation] = []
    forbidden = set(names)
    for path, tree in _trees(paths):
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            found: str | None = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found = node.name if node.name in forbidden else None
            elif isinstance(node, ast.Name) and node.id in forbidden:
                parent = parents.get(node)
                if not isinstance(parent, ast.Attribute) or parent.value is not node:
                    found = node.id
            elif isinstance(node, ast.Attribute) and node.attr in forbidden:
                found = node.attr
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    local_name = alias.asname or alias.name.rsplit(".", 1)[-1]
                    if local_name in forbidden:
                        violations.append(
                            SourceViolation(
                                "removed-legacy-api", path, node.lineno, local_name,
                                "Removed legacy APIs must not be restored or referenced.",
                            )
                        )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                parent = parents.get(node)
                if node.value in forbidden and isinstance(parent, (ast.List, ast.Tuple, ast.Set)):
                    for statement in tree.body:
                        if (
                            isinstance(statement, ast.Assign)
                            and statement.value is parent
                            and any(isinstance(target, ast.Name) and target.id == "__all__" for target in statement.targets)
                        ):
                            found = node.value
                            break
            if found is not None:
                violations.append(
                    SourceViolation(
                        "removed-legacy-api", path, node.lineno, found,
                        "Removed legacy APIs must not be restored or referenced.",
                    )
                )
    unique = {(item.source_path, item.line_number, item.qualified_name): item for item in violations}
    return tuple(sorted(unique.values(), key=lambda item: (str(item.source_path), item.line_number, item.qualified_name)))
