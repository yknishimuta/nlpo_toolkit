# tests/test_outputs.py
from __future__ import annotations

from collections import Counter
from pathlib import Path

from nlpo_toolkit.corpus_analysis.artifacts.models import ArtifactKind, PlannedArtifact
from nlpo_toolkit.corpus_analysis.artifacts.writers.frequency import write_frequency_artifact
from nlpo_toolkit.corpus_analysis.reporting.environment import collect_runtime_environment


def test_write_frequency_csv_writes_header_and_sorts(tmp_path: Path) -> None:
    out = tmp_path / "freq.csv"
    freq = Counter({"b": 1, "a": 2, "c": 2})

    write_frequency_artifact(PlannedArtifact(ArtifactKind.FREQUENCY, out, group="g"), counter=freq, header=("lemma", "count"))

    text = out.read_text(encoding="utf-8").splitlines()
    assert text[0] == "lemma,count"

    # count desc, then key asc => a(2), c(2), b(1)
    assert text[1:] == ["a,2", "c,2", "b,1"]


def test_collect_runtime_environment_does_not_crash_without_git(monkeypatch, tmp_path: Path) -> None:
    import subprocess as _subprocess

    def _boom(*args, **kwargs):
        raise _subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(_subprocess, "check_output", _boom)

    env = collect_runtime_environment(tmp_path)

    assert env.python_version
    assert env.git_commit is None
