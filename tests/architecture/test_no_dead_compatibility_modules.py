from pathlib import Path


REMOVED_MODULES = (
    Path("nlpo_toolkit/corpus_analysis/compose.py"),
    Path("nlpo_toolkit/corpus_analysis/text_prep.py"),
    Path("nlpo_toolkit/corpus_analysis/counters.py"),
    Path("nlpo_toolkit/corpus_analysis/nlp_hooks.py"),
)


def test_removed_modules_do_not_return() -> None:
    assert [path for path in REMOVED_MODULES if path.exists()] == []


def test_io_utils_has_only_active_helpers() -> None:
    import nlpo_toolkit.corpus_analysis.io_utils as io_utils

    assert hasattr(io_utils, "expand_globs")
    assert hasattr(io_utils, "read_concat")
    assert not hasattr(io_utils, "save_counter_csv")
    assert not hasattr(io_utils, "write_summary")
