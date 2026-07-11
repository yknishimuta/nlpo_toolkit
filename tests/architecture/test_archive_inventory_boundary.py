import ast
from pathlib import Path


ARCHIVE = Path("nlpo_toolkit/corpus_analysis/archive.py")


def test_archive_does_not_import_or_call_inventory_discovery_helpers() -> None:
    source = ARCHIVE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "expand_globs",
        "expand_cleaned_dir_placeholders",
        "resolve_cleaner_output_dir",
        "_collect_group_files",
        "_collect_input_files",
        "_collect_cleaned_files",
        "_collect_output_files",
        "_collect_trace_files",
        "_trace_base_path",
    }
    assert forbidden.isdisjoint(source.split())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"glob", "rglob"}


def test_archive_does_not_parse_run_metadata_or_config_schema() -> None:
    source = ARCHIVE.read_text(encoding="utf-8")
    assert "run_meta.json" not in source
    for fragment in (
        "config.groups",
        "config.preprocess",
        "config.trace",
        "config.artifacts",
        "config.dictcheck",
        "config.filters",
    ):
        assert fragment not in source
