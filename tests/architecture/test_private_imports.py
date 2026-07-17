from .support.rules import format_violations
from .support.source_checks import find_cross_module_private_imports


def test_production_has_no_cross_module_private_imports(production_paths) -> None:
    violations = find_cross_module_private_imports(production_paths)
    assert not violations, format_violations(violations)

