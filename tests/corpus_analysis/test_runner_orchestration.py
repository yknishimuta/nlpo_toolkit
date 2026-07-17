from __future__ import annotations

from tests.corpus_analysis.fake_nlp import corpus_request

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from nlpo_toolkit.nlp.contracts import BuiltNLPBackend, NLPBackendInfo
from nlpo_toolkit.corpus_analysis.config import ensure_app_config
from tests.corpus_analysis.fake_nlp import runner_dependencies

import nlpo_toolkit.corpus_analysis.runner as runner_mod


def test_run_orchestrates_steps_in_order(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    context = SimpleNamespace()
    analysis = SimpleNamespace()
    partitions = SimpleNamespace(exit_code=7)
    comparisons = SimpleNamespace()

    monkeypatch.setattr(
        runner_mod.runtime,
        "prepare_run_context",
        lambda *_args, **_kwargs: calls.append("prepare") or context,
    )
    monkeypatch.setattr(
        runner_mod,
        "analyze_corpora",
        lambda ctx, **_kwargs: calls.append("analyze") or analysis,
    )
    monkeypatch.setattr(
        runner_mod.post_analysis,
        "execute_partition_validations",
        lambda **kwargs: calls.append("partitions") or partitions,
    )
    monkeypatch.setattr(
        runner_mod.post_analysis,
        "execute_group_comparisons",
        lambda **kwargs: calls.append("comparisons") or comparisons,
    )
    monkeypatch.setattr(
        runner_mod,
        "build_run_report",
        lambda **kwargs: calls.append("report") or SimpleNamespace(),
    )
    monkeypatch.setattr(
        runner_mod,
        "build_count_run_result",
        lambda **kwargs: SimpleNamespace(exit_code=partitions.exit_code),
    )

    dependencies = runner_dependencies(
        lambda _path: ensure_app_config(
            {"groups": {"text": {"files": ["input/*.txt"]}}}
        ),
        lambda _config: BuiltNLPBackend(
            backend=object(),
            info=NLPBackendInfo(name="fake", language="la"),
        ),
    )
    dependencies = replace(
        dependencies,
        publication=replace(
            dependencies.publication,
            run_report=lambda _request: calls.append("publish"),
        ),
    )
    rc = runner_mod.run(
        corpus_request(tmp_path, tmp_path / "config.yml"),
        dependencies=dependencies,
    )

    assert calls == [
        "prepare", "analyze", "partitions", "comparisons", "report", "publish"
    ]
    assert rc.exit_code == 7
