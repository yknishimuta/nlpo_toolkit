from __future__ import annotations

from tests.corpus_analysis.fake_nlp import corpus_request

import csv
import json
from pathlib import Path

import pytest

from nlpo_toolkit.nlp.contracts import BuiltNLPBackend, NLPBackendInfo
from nlpo_toolkit.corpus_analysis import runner as runner_mod
from nlpo_toolkit.corpus_analysis.config import load_config
from nlpo_toolkit.corpus_analysis.token_artifact import read_token_records
from nlpo_toolkit.nlp.contracts import NLPDocument, NLPSentence, NLPToken
from tests.corpus_analysis.fake_nlp import runner_dependencies


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str) -> NLPDocument:
        self.calls.append(text)
        return NLPDocument(
            sentences=[
                NLPSentence(
                    text="Puella amat rosam .",
                    tokens=[
                        NLPToken("Puella", "puella", "NOUN", 0, 6),
                        NLPToken("amat", "amo", "VERB", 7, 11),
                        NLPToken("rosam", "rosa", "NOUN", 12, 17),
                        NLPToken(".", ".", "PUNCT", 18, 19),
                    ],
                )
            ],
            text=text,
        )


def _run(tmp_path: Path, config_text: str, backend: FakeBackend) -> int:
    (tmp_path / "input").mkdir(exist_ok=True)
    (tmp_path / "input" / "text.txt").write_text("Puella amat rosam .", encoding="utf-8")
    config_path = tmp_path / "groups.config.yml"
    config_path.write_text(config_text, encoding="utf-8")
    return runner_mod.run(
        corpus_request(tmp_path, config_path),
        dependencies=runner_dependencies(
            load_config,
            lambda config: BuiltNLPBackend(
                backend=backend,
                info=NLPBackendInfo(name="fake", language=config.language, package="package_a"),
            ),
        ),
    )


def _read_frequency(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    key_column = "lemma" if "lemma" in rows[0] else "word"
    return {row[key_column]: int(row["count"] if "count" in row else row["frequency"]) for row in rows}


def test_token_artifact_records_all_tokens_and_frequency_counts_included(tmp_path: Path) -> None:
    backend = FakeBackend()
    config = """
groups:
  text: {files: [input/text.txt]}
artifacts:
  tokens:
    enabled: true
    path: output/tokens.tsv
"""

    assert _run(tmp_path, config, backend).exit_code == 0
    assert len(backend.calls) == 1

    token_path = tmp_path / "output" / "tokens.tsv"
    records = list(read_token_records(token_path))
    assert [record.token for record in records] == ["Puella", "amat", "rosam", "."]
    assert [record.included for record in records] == [True, False, True, False]
    assert records[1].exclusion_reason == "upos_not_targeted"
    assert records[3].exclusion_reason == "upos_not_targeted"
    assert _read_frequency(tmp_path / "output" / "frequency_text.csv") == {
        "puella": 1,
        "rosa": 1,
    }

    meta = json.loads((tmp_path / "output" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["token_artifacts"][0]["group"] == "text"
    generated = "\n".join(meta["generated_outputs"])
    assert "tokens.tsv" in generated
    assert "tokens.meta.json" in generated


def test_diagnostic_trace_filters_but_token_artifact_remains_complete(tmp_path: Path) -> None:
    backend = FakeBackend()
    config = """
groups:
  text: {files: [input/text.txt]}
trace:
  enabled: true
  path: output/trace.tsv
  max_rows: 1
  only_keys: [puella]
artifacts:
  tokens:
    enabled: true
    path: output/tokens.tsv
"""

    assert _run(tmp_path, config, backend).exit_code == 0
    assert len(backend.calls) == 1
    assert len(list(read_token_records(tmp_path / "output" / "tokens.tsv"))) == 4

    trace_text = (tmp_path / "output" / "trace.tsv").read_text(encoding="utf-8")
    assert "Puella" in trace_text
    with (tmp_path / "output" / "trace.tsv").open("r", encoding="utf-8", newline="") as f:
        trace_rows = list(csv.DictReader(f, delimiter="\t"))
    assert [row["token"] for row in trace_rows] == ["Puella"]
    assert "trace stopped" not in trace_text


def test_token_artifact_and_trace_path_collision_is_rejected(tmp_path: Path) -> None:
    backend = FakeBackend()
    config = """
groups:
  text: {files: [input/text.txt]}
trace:
  enabled: true
  path: output/shared.tsv
artifacts:
  tokens:
    enabled: true
    path: output/shared.tsv
"""

    with pytest.raises(ValueError, match="Token artifact path and diagnostic trace path"):
        _run(tmp_path, config, backend)
