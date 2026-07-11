from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import nlpo_toolkit.corpus_analysis.runner as runner_mod
from nlpo_toolkit.corpus_analysis.config import load_config
from nlpo_toolkit.corpus_analysis.token_artifact import read_token_records
from tests.corpus_analysis.fake_nlp import FakeNLPBackend, fake_backend_factory, runner_dependencies


TOKENS = (
    ("Rosa", "rosa", "NOUN"),
    ("Marcus", "marcus", "PROPN"),
    ("xiv", "xiv", "NOUN"),
    ("a", "a", "NOUN"),
)


def _write_project(
    project_root: Path,
    *,
    cache_enabled: bool,
    artifact_enabled: bool,
    trace_enabled: bool,
) -> Path:
    (project_root / "input").mkdir(exist_ok=True)
    (project_root / "input" / "text.txt").write_text("Rosa Marcus xiv a", encoding="utf-8")
    config_path = project_root / "groups.config.yml"
    config_path.write_text(
        "\n".join(
            [
                "out_dir: output",
                "groups:",
                "  text: {files: [input/text.txt]}",
                "filters:",
                "  upos_targets: [NOUN]",
                "  min_token_length: 2",
                "  drop_roman_numerals: true",
                "analysis_cache:",
                f"  enabled: {str(cache_enabled).lower()}",
                "  dir: .analysis_cache",
                "artifacts:",
                "  tokens:",
                f"    enabled: {str(artifact_enabled).lower()}",
                "    path: output/tokens.tsv",
                "trace:",
                f"  enabled: {str(trace_enabled).lower()}",
                "  path: output/trace.tsv",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _run(project_root: Path, config_path: Path, backend: FakeNLPBackend) -> None:
    rc = runner_mod.run(
        project_root=project_root,
        config_path=config_path,
        dependencies=runner_dependencies(
            load_config,
            fake_backend_factory(backend=backend),
        ),
    )
    assert rc.exit_code == 0


def _frequency(project_root: Path) -> dict[str, int]:
    with (project_root / "output" / "frequency_text.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))
    return {row["lemma"]: int(row["count"]) for row in rows}


@pytest.mark.parametrize(
    ("cache_enabled", "artifact_enabled", "trace_enabled"),
    [
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, True, True),
    ],
)
def test_count_result_is_independent_of_optional_outputs(
    tmp_path: Path,
    cache_enabled: bool,
    artifact_enabled: bool,
    trace_enabled: bool,
) -> None:
    config_path = _write_project(
        tmp_path,
        cache_enabled=cache_enabled,
        artifact_enabled=artifact_enabled,
        trace_enabled=trace_enabled,
    )
    backend = FakeNLPBackend(tokens=TOKENS)

    _run(tmp_path, config_path, backend)

    assert _frequency(tmp_path) == {"rosa": 1}
    assert len(backend.calls) == 1

    artifact_path = tmp_path / "output" / "tokens.tsv"
    if artifact_enabled:
        records = list(read_token_records(artifact_path))
        by_token = {record.token: record for record in records}
        assert by_token["Rosa"].included is True
        assert by_token["Rosa"].analysis_key == "rosa"
        assert by_token["Marcus"].exclusion_reason == "upos_not_targeted"
        assert by_token["xiv"].exclusion_reason == "roman_numeral"
        assert by_token["a"].exclusion_reason == "too_short"
    else:
        assert not artifact_path.exists()

    trace_path = tmp_path / "output" / "trace.tsv"
    if trace_enabled:
        rows = list(csv.DictReader(trace_path.open(encoding="utf-8"), delimiter="\t"))
        assert [row["token"] for row in rows] == ["Rosa"]
    else:
        assert not trace_path.exists()


def test_cache_hit_regenerates_artifact_and_trace_without_nlp(tmp_path: Path) -> None:
    config_path = _write_project(
        tmp_path,
        cache_enabled=True,
        artifact_enabled=True,
        trace_enabled=True,
    )
    backend = FakeNLPBackend(tokens=TOKENS)

    _run(tmp_path, config_path, backend)
    assert len(backend.calls) == 1
    first_meta = json.loads((tmp_path / "output" / "run_meta.json").read_text(encoding="utf-8"))
    assert first_meta["analysis_cache"]["misses"] == 1

    (tmp_path / "output" / "tokens.tsv").unlink()
    (tmp_path / "output" / "tokens.meta.json").unlink()
    (tmp_path / "output" / "trace.tsv").unlink()

    _run(tmp_path, config_path, backend)

    assert len(backend.calls) == 1
    assert _frequency(tmp_path) == {"rosa": 1}
    assert (tmp_path / "output" / "tokens.tsv").exists()
    assert (tmp_path / "output" / "trace.tsv").exists()
    second_meta = json.loads((tmp_path / "output" / "run_meta.json").read_text(encoding="utf-8"))
    assert second_meta["analysis_cache"]["hits"] == 1
