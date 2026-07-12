from pathlib import Path
import inspect

from nlpo_toolkit.corpus_analysis import features
from nlpo_toolkit.corpus_analysis.cli import count as count_cli
from nlpo_toolkit.corpus_analysis.cli import features as features_cli
from nlpo_toolkit.corpus_analysis.runner import run


def test_cli_has_no_monkeypatch_detection() -> None:
    paths = (
        Path("nlpo_toolkit/corpus_analysis/cli/count.py"),
        Path("nlpo_toolkit/corpus_analysis/cli/features.py"),
    )
    forbidden = (
        "_DEFAULT_BUILD_PIPELINE",
        "_DEFAULT_BUILD_SENTENCE_SPLITTER",
        "legacy_build_pipeline",
        "legacy_sentence_splitter",
        " is not _DEFAULT_",
    )
    offenders = [
        (str(path), fragment)
        for path in paths
        for fragment in forbidden
        if fragment in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_runner_and_features_have_no_legacy_pipeline_arguments() -> None:
    runner_parameters = inspect.signature(run).parameters
    feature_parameters = inspect.signature(
        features.execute_feature_command
    ).parameters
    assert "build_pipeline_fn" not in runner_parameters
    assert "build_sentence_splitter_fn" not in runner_parameters
    assert "build_pipeline_fn" not in feature_parameters
    assert "dependencies" in runner_parameters
    assert "dependencies" in feature_parameters


def test_cli_has_no_pipeline_wrapper_attributes() -> None:
    assert not hasattr(count_cli, "build_pipeline")
    assert not hasattr(count_cli, "build_sentence_splitter")
    assert not hasattr(features_cli, "build_pipeline")
