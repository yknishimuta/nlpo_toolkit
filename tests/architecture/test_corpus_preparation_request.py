from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

from nlpo_toolkit.corpus_analysis.count_command import CountRequest
from nlpo_toolkit.corpus_analysis.features import FeatureRequest
from nlpo_toolkit.corpus_analysis.ngram import ConfigNgramRequest
from nlpo_toolkit.corpus_analysis.requests import CorpusPreparationRequest


SHARED_FIELDS = {
    "project_root",
    "config_path",
    "group_by_file",
    "auto_single_cleaned",
    "error_on_empty_group",
}


def test_command_requests_do_not_redeclare_corpus_preparation_fields() -> None:
    assert {field.name for field in fields(CorpusPreparationRequest)} == {
        "project_root",
        "config_path",
        "grouping_override",
        "error_on_empty_group",
    }
    for request_type in (CountRequest, FeatureRequest, ConfigNgramRequest):
        names = {field.name for field in fields(request_type)}
        assert "corpus" in names
        assert not names & SHARED_FIELDS
    assert "dry_run" not in {field.name for field in fields(CountRequest)}
    assert "field" not in {field.name for field in fields(ConfigNgramRequest)}


def test_command_services_pass_composed_request_without_expanding_old_kwargs() -> None:
    paths = (
        Path("nlpo_toolkit/corpus_analysis/count_command.py"),
        Path("nlpo_toolkit/corpus_analysis/features.py"),
        Path("nlpo_toolkit/corpus_analysis/ngram.py"),
    )
    forbidden_keywords = {
        "project_root",
        "config_path",
        "group_by_file",
        "auto_single_cleaned",
        "error_on_empty_group",
    }
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            assert not forbidden_keywords & {
                keyword.arg for keyword in call.keywords if keyword.arg is not None
            }, path


def test_cli_commands_use_one_canonical_request_factory() -> None:
    for name in ("count", "features", "ngram"):
        source = Path(f"nlpo_toolkit/corpus_analysis/cli/{name}.py").read_text(
            encoding="utf-8"
        )
        assert source.count("build_corpus_preparation_request(") == 1


def test_argparse_namespace_is_confined_to_cli_layer() -> None:
    offenders: list[Path] = []
    root = Path("nlpo_toolkit/corpus_analysis")
    for path in root.rglob("*.py"):
        if "cli" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "Namespace":
                offenders.append(path)
    assert offenders == []
