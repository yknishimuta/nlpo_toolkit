from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from nlpo_toolkit.corpus_analysis.config import ConfigError, load_config, parse_config


def test_validation_errors_include_all_dotted_locations_without_documentation_urls() -> None:
    with pytest.raises(ConfigError) as caught:
        parse_config(
            {
                "groups": {"a": {"files": ["a.txt"], "unknown": True}},
                "filters": {"min_token_length": -1},
                "analysis_cache": {"dir": 3},
            }
        )
    message = str(caught.value)
    assert "groups.a.unknown" in message
    assert "filters.min_token_length" in message
    assert "analysis_cache.dir" in message
    assert "https://errors.pydantic.dev" not in message
    assert not isinstance(caught.value, ValidationError)


@pytest.mark.parametrize("suffix", [".yml", ".yaml"])
def test_yaml_extensions_are_supported(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"config{suffix}"
    path.write_text("groups:\n  a: {files: [a.txt]}\n", encoding="utf-8")
    assert load_config(path).groups["a"].files == ("a.txt",)
