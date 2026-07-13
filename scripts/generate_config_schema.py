from __future__ import annotations

import argparse
import json
from pathlib import Path

from nlpo_toolkit.corpus_analysis.config import AppConfig


DEFAULT_OUTPUT = Path("config/groups.config.schema.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the public groups config JSON Schema from AppConfig."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    schema = AppConfig.model_json_schema(by_alias=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
