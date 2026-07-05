from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

from .config import load_config
from .nlp_hooks import (
    build_pipeline,
    build_sentence_splitter,
    count_group,
    render_stanza_package_table,
)
from .runner import run

try:
    from nlpo_toolkit.latin.cleaners import run_clean_corpus as clean_mod
except Exception:
    clean_mod = SimpleNamespace(main=lambda argv: 0)


def run_count_vocabula(
    *,
    project_root: Path,
    config_path: Path,
    group_by_file: bool = False,
) -> int:
    return run(
        project_root=project_root,
        config_path=config_path,
        group_by_file=group_by_file,
        load_config_fn=load_config,
        clean_mod=clean_mod,
        build_pipeline_fn=build_pipeline,
        build_sentence_splitter_fn=build_sentence_splitter,
        count_group_fn=count_group,
        render_stanza_package_table_fn=render_stanza_package_table,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nlpo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("count-vocabula", "count"):
        count_parser = subparsers.add_parser(name)
        count_parser.add_argument(
            "--project-root",
            type=Path,
            default=Path.cwd(),
            help="Project root used to resolve relative paths in the config.",
        )
        count_parser.add_argument(
            "--config",
            type=Path,
            default=None,
            help="YAML config path. Defaults to <project-root>/config/groups.config.yml.",
        )
        count_parser.add_argument(
            "--group-by-file",
            action="store_true",
            help="Write one frequency CSV per input file instead of one CSV per configured group.",
        )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    config_path = args.config
    if config_path is None:
        config_path = project_root / "config" / "groups.config.yml"
    elif not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    if args.command in {"count-vocabula", "count"}:
        return run_count_vocabula(
            project_root=project_root,
            config_path=config_path,
            group_by_file=bool(args.group_by_file),
        )

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
