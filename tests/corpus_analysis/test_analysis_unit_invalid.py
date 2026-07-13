from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.runner import run
from tests.corpus_analysis.fake_nlp import (
    corpus_request,
    fake_backend_factory,
    runner_dependencies,
)


def test_analysis_unit_invalid_raises_value_error(tmp_path):
    script_dir = tmp_path / "runner_dir"
    (script_dir / "config").mkdir(parents=True, exist_ok=True)
    config_path = script_dir / "config" / "groups.config.yml"
    config_path.write_text("dummy", encoding="utf-8")

    data_dir = script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "t1.txt").write_text("Puella rosam amat.\n", encoding="utf-8")

    out_dir = script_dir / "output"

    cfg = {
        "groups": {"text": {"files": [str(data_dir / "*.txt")]}},
        "out_dir": str(out_dir),
        "analysis_unit": "LEMMAAAA",  # invalid on purpose
        "nlp": {
            "language": "la",
            "stanza_package": "perseus",
            "cpu_only": True,
        },
        "dictcheck": {"enabled": False},
    }

    def fake_load_config(path: Path):
        assert path.name == "groups.config.yml"
        return cfg


    dependencies = runner_dependencies(
        fake_load_config,
        fake_backend_factory([("x", "x", "NOUN")]),
    )
    with pytest.raises(ValueError, match=r"analysis_unit"):
        run(
            corpus_request(script_dir, config_path),
            dependencies=dependencies,
        )
