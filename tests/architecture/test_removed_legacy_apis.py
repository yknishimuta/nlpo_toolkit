from .support.rules import format_violations
from .support.source_checks import find_forbidden_identifiers


REMOVED_MODULES = frozenset({
    "nlpo_toolkit.corpus_analysis.archive_types",
    "nlpo_toolkit.corpus_analysis.dictcheck",
    "nlpo_toolkit.corpus_analysis.preprocess",
    "nlpo_toolkit.nlp.normalization",
})

REMOVED_IDENTIFIERS = frozenset({
    "DEFAULT_LIGATURE_MAP",
    "counter_from_token_records",
    "iter_token_records",
    "normalize_token",
    "ref_tag_counter",
    "ref_tag_detector",
})


def test_removed_legacy_modules_are_not_in_the_production_graph(production_graph) -> None:
    restored = sorted(REMOVED_MODULES & production_graph.modules)
    assert not restored, "Removed legacy modules were restored:\n" + "\n".join(restored)


def test_removed_legacy_identifiers_are_not_in_production(production_paths) -> None:
    violations = find_forbidden_identifiers(production_paths, names=REMOVED_IDENTIFIERS)
    assert not violations, format_violations(violations)
