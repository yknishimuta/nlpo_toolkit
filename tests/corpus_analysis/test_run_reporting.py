from __future__ import annotations

def test_run_reporting_functions_have_canonical_module() -> None:
    from nlpo_toolkit.corpus_analysis.run_reporting import build_final_run_metadata

    assert build_final_run_metadata.__module__.endswith(".run_reporting")
