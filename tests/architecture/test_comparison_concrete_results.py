from pathlib import Path

from .support.rules import format_violations
from .support.source_checks import find_forbidden_identifiers, find_generic_class_bases


RESULT_CLASSES = frozenset({
    "PairwiseComparisonResult",
    "MultiComparisonResult",
    "ConfiguredComparisonResult",
    "CsvPairComparisonResult",
    "CsvMultiComparisonResult",
})


def _comparison_paths(production_paths) -> tuple[Path, ...]:
    marker = f"nlpo_toolkit{Path('/')}comparison{Path('/')}"
    return tuple(path for path in production_paths if marker in str(path))


def test_comparison_result_type_variables_are_not_reintroduced(production_paths) -> None:
    violations = find_forbidden_identifiers(
        _comparison_paths(production_paths),
        names={"TFrequencyTable", "TComparisonSpec"},
    )
    assert not violations, format_violations(violations)


def test_comparison_results_do_not_inherit_generic(production_paths) -> None:
    violations = find_generic_class_bases(
        _comparison_paths(production_paths),
        class_names=RESULT_CLASSES,
    )
    assert not violations, format_violations(violations)
