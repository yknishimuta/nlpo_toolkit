from __future__ import annotations

from pathlib import Path
import yaml


def load_clean_config(path: str | Path) -> dict:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    if "kind" not in data:
        raise ValueError(f"'kind' is required in clean config YAML: {path}")
    if "input" not in data or "output" not in data:
        raise ValueError(f"'input' and 'output' are required in clean config YAML: {path}")

    return data
