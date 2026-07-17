from .policy import PACKAGE_CYCLE_GROUPS
from .support.module_graph import collapse_graph, cycle_diagnostics, find_cycles


def test_production_modules_are_acyclic(production_graph) -> None:
    cycles = find_cycles(production_graph)
    assert not cycles, cycle_diagnostics(production_graph, cycles)


def test_production_packages_are_acyclic(production_graph) -> None:
    graph = collapse_graph(production_graph, PACKAGE_CYCLE_GROUPS)
    cycles = find_cycles(graph)
    assert not cycles, cycle_diagnostics(graph, cycles)

