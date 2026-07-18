from __future__ import annotations

import argparse
from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.cli.common import (
    build_corpus_preparation_request,
)
from nlpo_toolkit.corpus_analysis.cli.main import build_parser
from nlpo_toolkit.corpus_analysis.count_command import CountRequest
from nlpo_toolkit.corpus_analysis.features.models import FeatureRequest
from nlpo_toolkit.corpus_analysis.ngram import ConfigNgramRequest, TokenNgramRequest
from nlpo_toolkit.corpus_analysis.requests import CorpusPreparationRequest


def test_corpus_preparation_request_is_frozen_and_has_canonical_defaults(
    tmp_path: Path,
) -> None:
    request = CorpusPreparationRequest(tmp_path, tmp_path / "groups.yml")
    assert is_dataclass(request)
    assert request.project_root == tmp_path
    assert request.config_path == tmp_path / "groups.yml"
    assert request.grouping_override is None
    assert request.error_on_empty_group is False
    with pytest.raises(FrozenInstanceError):
        request.grouping_override = "per_file"  # type: ignore[misc]


@pytest.mark.parametrize("override", ["per_file", "auto_single_cleaned"])
def test_corpus_preparation_request_accepts_each_grouping_override(
    tmp_path: Path, override: str
) -> None:
    request = CorpusPreparationRequest(
        tmp_path,
        tmp_path / "groups.yml",
        grouping_override=override,  # type: ignore[arg-type]
        error_on_empty_group=True,
    )
    assert request.grouping_override == override
    assert request.error_on_empty_group is True


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"project_root": "root"}, TypeError),
        ({"config_path": "config.yml"}, TypeError),
        ({"grouping_override": "unknown"}, ValueError),
        ({"error_on_empty_group": 1}, TypeError),
    ],
)
def test_corpus_preparation_request_rejects_invalid_runtime_values(
    tmp_path: Path, kwargs: dict[str, object], error: type[Exception]
) -> None:
    values: dict[str, object] = {
        "project_root": tmp_path,
        "config_path": tmp_path / "groups.yml",
        **kwargs,
    }
    with pytest.raises(error):
        CorpusPreparationRequest(**values)  # type: ignore[arg-type]


def test_command_requests_compose_corpus_without_duplicate_fields() -> None:
    assert {field.name for field in fields(CountRequest)} == {
        "corpus",
        "command_line",
        "archive_run",
        "run_name",
        "runs_dir",
        "include_input",
        "include_cleaned",
    }
    assert {field.name for field in fields(FeatureRequest)} == {
        "corpus",
        "field",
        "mfw",
        "include_upos",
        "include_basic",
        "sampling",
        "lexical_diversity",
    }
    assert {field.name for field in fields(ConfigNgramRequest)} == {
        "corpus",
        "n",
        "by_group",
        "min_count",
        "top",
    }
    assert "field" in {field.name for field in fields(TokenNgramRequest)}


@pytest.mark.parametrize(
    ("group_by_file", "auto_single_cleaned", "expected"),
    [
        (False, False, None),
        (True, False, "per_file"),
        (False, True, "auto_single_cleaned"),
    ],
)
def test_cli_factory_resolves_paths_and_maps_one_grouping_override(
    tmp_path: Path,
    group_by_file: bool,
    auto_single_cleaned: bool,
    expected: str | None,
) -> None:
    args = argparse.Namespace(
        project_root=tmp_path / ".",
        config=Path("config/custom.yml"),
        group_by_file=group_by_file,
        auto_single_cleaned=auto_single_cleaned,
        error_on_empty_group=True,
    )
    request = build_corpus_preparation_request(args)
    assert request.project_root == tmp_path.resolve()
    assert request.config_path == (tmp_path / "config/custom.yml").resolve()
    assert request.grouping_override == expected
    assert request.error_on_empty_group is True


@pytest.mark.parametrize(
    "argv",
    [
        ["count", "--group-by-file", "--auto-single-cleaned"],
        ["features", "--group-by-file", "--auto-single-cleaned"],
        [
            "ngram",
            "--config",
            "groups.yml",
            "--group-by-file",
            "--auto-single-cleaned",
        ],
    ],
)
def test_cli_rejects_multiple_grouping_overrides(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args(argv)
    assert caught.value.code == 2
