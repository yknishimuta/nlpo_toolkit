from __future__ import annotations

import ast
from pathlib import Path


ROOTS = (Path("nlpo_toolkit"), Path("tests"))


def _is_private_name(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _module_name(path: Path) -> str:
    return ".".join(path.with_suffix("").parts)


def _is_project_module(module: str | None) -> bool:
    return bool(module) and (
        module == "nlpo_toolkit"
        or module.startswith("nlpo_toolkit.")
        or module == "tests"
        or module.startswith("tests.")
    )


def test_no_cross_module_private_imports() -> None:
    violations: list[str] = []

    for root in ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            module_aliases: dict[str, str] = {}
            current_module = _module_name(path)

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module
                    if node.level:
                        # Relative imports inside this project are in scope even
                        # when ast does not resolve the absolute module name.
                        module_is_project = True
                    else:
                        module_is_project = _is_project_module(module)
                    if not module_is_project:
                        continue
                    for alias in node.names:
                        if _is_private_name(alias.name):
                            violations.append(
                                f"{path}:{node.lineno}: imports private symbol "
                                f"{alias.name!r} from {module!r}"
                            )
                    continue

                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if _is_project_module(alias.name):
                            module_aliases[alias.asname or alias.name.split(".")[0]] = alias.name

            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute):
                    continue
                if not _is_private_name(node.attr):
                    continue
                if isinstance(node.value, ast.Name) and node.value.id in module_aliases:
                    violations.append(
                        f"{path}:{node.lineno}: accesses private attribute "
                        f"{node.value.id}.{node.attr} from {module_aliases[node.value.id]!r}"
                    )

    assert violations == [], "\n".join(violations)
