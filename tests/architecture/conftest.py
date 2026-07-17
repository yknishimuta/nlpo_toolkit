from __future__ import annotations

from pathlib import Path

import pytest

from .support.module_graph import ModuleGraph, build_module_graph


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOT = PROJECT_ROOT / "nlpo_toolkit"


@pytest.fixture(scope="session")
def production_graph() -> ModuleGraph:
    if not PRODUCTION_ROOT.is_dir():
        raise RuntimeError(f"production package root is missing: {PRODUCTION_ROOT}")
    return build_module_graph(PRODUCTION_ROOT, package_name="nlpo_toolkit")


@pytest.fixture(scope="session")
def production_paths() -> tuple[Path, ...]:
    return tuple(sorted(PRODUCTION_ROOT.rglob("*.py")))

