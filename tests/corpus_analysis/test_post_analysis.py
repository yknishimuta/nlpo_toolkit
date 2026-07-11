from __future__ import annotations


def test_post_analysis_functions_have_canonical_module() -> None:
    from nlpo_toolkit.corpus_analysis.post_analysis import (
        execute_group_comparisons,
        execute_partition_validations,
    )

    assert execute_partition_validations.__module__.endswith(".post_analysis")
    assert execute_group_comparisons.__module__.endswith(".post_analysis")
