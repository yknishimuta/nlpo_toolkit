from .policy import DYNAMIC_IMPORT_ALLOWANCES
from .support.rules import matches_prefix


def test_non_literal_dynamic_imports_are_explicitly_allowed(production_graph) -> None:
    offenders = []
    for issue in production_graph.dynamic_import_issues:
        allowed = any(
            matches_prefix(issue.importer, item.module_prefix)
            for item in DYNAMIC_IMPORT_ALLOWANCES
        )
        if not allowed:
            offenders.append(issue)
    assert not offenders, "\n\n".join(str(item) for item in offenders)

