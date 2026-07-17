from pathlib import Path

from .support.rules import format_violations
from .support.source_checks import find_mutable_fields_in_frozen_models


def test_frozen_value_models_do_not_declare_mutable_collection_fields(
    production_paths,
) -> None:
    violations = find_mutable_fields_in_frozen_models(
        production_paths,
        package_root=Path("nlpo_toolkit"),
        package_name="nlpo_toolkit",
    )
    assert not violations, format_violations(violations)
