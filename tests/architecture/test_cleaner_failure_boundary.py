from pathlib import Path


def test_cli_has_no_silent_cleaner_fallback() -> None:
    paths = (
        Path("nlpo_toolkit/corpus_analysis/cli/count.py"),
        Path("nlpo_toolkit/corpus_analysis/cli/features.py"),
        Path("nlpo_toolkit/corpus_analysis/cli/ngram.py"),
    )
    forbidden = ("SimpleNamespace", "main=lambda argv: 0")
    offenders = [
        (str(path), fragment)
        for path in paths
        for fragment in forbidden
        if fragment in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_default_cleaner_service_import_is_lazy_and_centralized() -> None:
    assert not Path("nlpo_toolkit/corpus_analysis/cleaner_runtime.py").exists()
    composition = Path("nlpo_toolkit/corpus_analysis/composition.py").read_text(encoding="utf-8")
    assert "from nlpo_toolkit.latin.cleaners.service import execute_cleaner" in composition
    assert "run_clean_corpus" not in composition

    for path in Path("nlpo_toolkit/corpus_analysis/cli").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "latin.cleaners" not in source, path
