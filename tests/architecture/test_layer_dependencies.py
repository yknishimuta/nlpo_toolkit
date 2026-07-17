from .policy import DEPENDENCY_RULES
from .support.module_graph import find_forbidden_dependencies
from .support.rules import format_violations


def test_dependency_direction(production_graph) -> None:
    violations = find_forbidden_dependencies(production_graph, DEPENDENCY_RULES)
    assert not violations, format_violations(violations)

