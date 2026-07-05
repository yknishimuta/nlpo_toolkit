from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from nlpo_toolkit.count_vocabula.cli import clean_mod
from nlpo_toolkit.count_vocabula.config import load_config
from nlpo_toolkit.count_vocabula.nlp_hooks import (
    build_pipeline,
    build_sentence_splitter,
    count_group,
    render_stanza_package_table,
)
from nlpo_toolkit.count_vocabula.preprocess import (
    expand_cleaned_dir_placeholders as _expand_cleaned_dir_placeholders,
)
from nlpo_toolkit.count_vocabula.runner import run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="count_corpus_vocabula_local.py")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    config_path = args.config
    if config_path is None:
        config_path = project_root / "config" / "groups.config.yml"
    elif not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    return run(
        project_root=project_root,
        config_path=config_path,
        load_config_fn=load_config,
        clean_mod=clean_mod,
        build_pipeline_fn=build_pipeline,
        build_sentence_splitter_fn=build_sentence_splitter,
        count_group_fn=count_group,
        render_stanza_package_table_fn=render_stanza_package_table,
    )


if __name__ == "__main__":
    raise SystemExit(main())
