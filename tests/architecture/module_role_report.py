from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

from .policy import MODULE_ROLE_POLICIES
from .support.module_graph import build_module_graph
from .support.module_roles import ModuleRolePolicy, find_unclassified_modules


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOT = PROJECT_ROOT / "nlpo_toolkit"


def render_unclassified_modules_report(modules: Iterable[str]) -> str:
    ordered = tuple(sorted(set(modules)))
    if not ordered:
        return "No unclassified production modules."
    lines = (
        "Unclassified production modules:",
        *(f"- {module}" for module in ordered),
    )
    return "\n".join(lines)


def run_report(
    modules: Iterable[str],
    policies: tuple[ModuleRolePolicy, ...],
    *,
    stream: TextIO,
) -> int:
    unclassified = find_unclassified_modules(modules, policies)
    print(render_unclassified_modules_report(unclassified), file=stream)
    return 0


def main() -> int:
    graph = build_module_graph(PRODUCTION_ROOT, package_name="nlpo_toolkit")
    return run_report(graph.modules, MODULE_ROLE_POLICIES, stream=sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
