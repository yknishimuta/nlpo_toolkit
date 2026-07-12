from __future__ import annotations

from pathlib import Path

import pytest

from nlpo_toolkit.backends import BuiltNLPBackend, NLPBackendInfo
from nlpo_toolkit.corpus_analysis.config import ensure_app_config
from nlpo_toolkit.corpus_analysis.runtime import prepare_run_context
from tests.corpus_analysis.fake_nlp import runner_dependencies


def test_prepare_run_context_validates_plan_before_nlp_initialization(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy", encoding="utf-8")
    calls: list[object] = []

    def backend_factory(config):
        calls.append(config)
        return BuiltNLPBackend(
            backend=object(),
            info=NLPBackendInfo(name="fake", language="la"),
        )

    deps = runner_dependencies(
        lambda _path: ensure_app_config({
            "groups": {
                "a": {"files": ["input/a.txt"]},
                "b": {"files": ["input/missing.txt"]},
            },
            "comparisons": [{"name": "ab", "group_a": "a", "group_b": "b"}],
        }),
        backend_factory,
    )

    with pytest.raises(ValueError, match="comparison ab references empty group: a"):
        prepare_run_context(
            project_root=tmp_path,
            script_dir=None,
            config_path=config_path,
            group_by_file=None,
            auto_single_cleaned=False,
            error_on_empty_group=False,
            dependencies=deps,
        )

    assert calls == []
