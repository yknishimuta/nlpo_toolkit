from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "nlpo_toolkit/corpus_analysis"


def _imports(name: str) -> tuple[set[str], set[str]]:
    tree = ast.parse((PACKAGE / name).read_text(encoding="utf-8"))
    modules: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module)
            names.update(alias.name for alias in node.names)
    return modules, names


def test_analysis_modules_replace_pipeline() -> None:
    for name in (
        "analysis_execution.py",
        "analysis_orchestration.py",
    ):
        assert (PACKAGE / name).is_file()
    assert not (PACKAGE / "analysis_outputs.py").exists()
    assert (PACKAGE / "postprocessing/service.py").is_file()
    assert (PACKAGE / "artifacts/writers/group.py").is_file()
    assert not (PACKAGE / "analysis_pipeline.py").exists()


def test_orchestration_combines_execution_and_outputs_only_at_facade() -> None:
    modules, names = _imports("analysis_orchestration.py")
    assert "analysis_execution" in names
    assert "postprocess_group_counter" in names
    assert "write_group_artifacts" in names
    assert names.isdisjoint(
        {
            "evaluate_analysis_record",
            "TokenArtifactWriter",
            "DiagnosticTraceWriter",
            "write_frequency_csv",
            "open_or_compute_analysis_records",
        }
    )


def test_execution_boundary() -> None:
    modules, names = _imports("analysis_execution.py")
    assert not any(module.endswith("analysis_orchestration") for module in modules)
    assert names.isdisjoint(
        {"write_frequency_csv", "load_vocab", "GroupAnalysisResult", "AnalysisResults"}
    )


def test_postprocessing_boundary() -> None:
    source = (PACKAGE / "postprocessing/service.py").read_text(encoding="utf-8")
    assert "ArtifactPlan" not in source
    assert "csv" not in source


def test_production_has_no_analysis_pipeline_import_or_export() -> None:
    offenders: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        modules, names = _imports(str(path.relative_to(PACKAGE)))
        if any(module.endswith("analysis_pipeline") for module in modules):
            offenders.append(str(path.relative_to(ROOT)))
        if "analysis_pipeline" in names:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_runner_imports_canonical_orchestrator() -> None:
    modules, names = _imports("runner.py")
    assert "analysis_orchestration" in modules
    assert "analyze_corpora" in names


def test_analysis_modules_import_in_fresh_process() -> None:
    code = """
import nlpo_toolkit.corpus_analysis.analysis_execution
import nlpo_toolkit.corpus_analysis.postprocessing.service
import nlpo_toolkit.corpus_analysis.artifacts.writers.group
import nlpo_toolkit.corpus_analysis.analysis_orchestration
import nlpo_toolkit.corpus_analysis.runner
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
