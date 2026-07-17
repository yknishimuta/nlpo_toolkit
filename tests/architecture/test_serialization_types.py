from .policy import SERIALIZATION_BOUNDARY_MODULES
from .support.rules import ArchitectureViolation, format_violations, matches_prefix


def test_generic_serialization_types_stay_at_serialization_boundaries(production_graph) -> None:
    violations = tuple(
        ArchitectureViolation(
            "generic-values-only-at-serialization-boundary",
            edge.importer,
            edge.imported,
            edge.source_path,
            edge.line_number,
            "Generic JSON/YAML/CSV values are permitted only while parsing or rendering at an explicit serialization boundary.",
        )
        for edge in production_graph.edges
        if matches_prefix(edge.imported, "nlpo_toolkit.serialization.types")
        and not any(matches_prefix(edge.importer, prefix) for prefix in SERIALIZATION_BOUNDARY_MODULES)
    )
    assert not violations, format_violations(violations)
