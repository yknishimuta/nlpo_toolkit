from __future__ import annotations

import ast
import inspect
import io
from pathlib import Path

from nlpo_toolkit.comparison.cli_service import execute_compare_command
from nlpo_toolkit.corpus_analysis.cache import clear_cache
from nlpo_toolkit.corpus_analysis.cli.common import CLIContext
from nlpo_toolkit.corpus_analysis.concordance import build_concordance
from nlpo_toolkit.corpus_analysis.count_command import execute_count_command
from nlpo_toolkit.corpus_analysis.dry_run import execute_dry_run
from nlpo_toolkit.corpus_analysis.features.service import execute_feature_command
from nlpo_toolkit.corpus_analysis.ngram import (
    execute_config_ngram_command,
    execute_token_ngram_command,
)


APPLICATION_MODULES = (
    Path("nlpo_toolkit/corpus_analysis/count_command.py"),
    Path("nlpo_toolkit/corpus_analysis/dry_run.py"),
    Path("nlpo_toolkit/corpus_analysis/features/service.py"),
    Path("nlpo_toolkit/corpus_analysis/ngram.py"),
    Path("nlpo_toolkit/corpus_analysis/concordance.py"),
    Path("nlpo_toolkit/corpus_analysis/cache.py"),
    Path("nlpo_toolkit/comparison/cli_service.py"),
)
APPLICATION_SERVICES = (
    execute_count_command,
    execute_dry_run,
    execute_feature_command,
    execute_config_ngram_command,
    execute_token_ngram_command,
    build_concordance,
    execute_compare_command,
    clear_cache,
)


def test_application_modules_do_not_use_cli_or_terminal_apis() -> None:
    offenders: list[str] = []
    for path in APPLICATION_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "print":
                    offenders.append(f"{path}:{node.lineno}: print")
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "sys" and node.attr in {"stdout", "stderr"}:
                    offenders.append(f"{path}:{node.lineno}: sys.{node.attr}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "argparse" or ".cli" in alias.name:
                        offenders.append(f"{path}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "argparse" or ".cli" in module:
                    offenders.append(f"{path}:{node.lineno}: from {module}")
            elif isinstance(node, ast.Name) and node.id == "CLIContext":
                offenders.append(f"{path}:{node.lineno}: CLIContext")
    assert offenders == []


def test_application_services_do_not_return_cli_exit_codes() -> None:
    for function in APPLICATION_SERVICES:
        annotation = inspect.signature(function).return_annotation
        assert annotation not in {int, "int"}, function.__name__


def test_cli_context_has_distinct_explicit_streams() -> None:
    context = CLIContext(
        argv=("nlpo", "count"),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert context.stdout is not context.stderr
