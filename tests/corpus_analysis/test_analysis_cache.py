from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from nlpo_toolkit.backends import BuiltNLPBackend, NLPBackendInfo
from nlpo_toolkit.corpus_analysis.analysis_cache import (
    AnalysisFingerprint,
    build_analysis_cache_key,
    cache_lock_path,
    cache_metadata_path,
    get_or_compute_analysis_records,
    prepared_text_sha256,
    prune_analysis_cache,
    read_analysis_records,
)
from nlpo_toolkit.corpus_analysis.analysis_policy import DEFAULT_ANALYSIS_EXTRACTION_POLICY
from nlpo_toolkit.corpus_analysis.analysis_policy import AnalysisExtractionPolicy
from nlpo_toolkit.corpus_analysis import cache_storage
from nlpo_toolkit.corpus_analysis.config import load_config
from nlpo_toolkit.corpus_analysis import runner as runner_mod
from nlpo_toolkit.corpus_analysis.analysis_records import NLPAnalysisRecord
from nlpo_toolkit.models import NLPDocument, NLPSentence, NLPToken
from tests.corpus_analysis.fake_nlp import runner_dependencies


def _analysis_record(**overrides) -> NLPAnalysisRecord:
    data = {
        "chunk_index": 0,
        "sentence_index": 0,
        "token_index": 0,
        "global_token_index": 0,
        "char_start_in_chunk": 0,
        "char_end_in_chunk": 5,
        "char_start_in_text": 0,
        "char_end_in_text": 5,
        "sentence": "Rosa amat.",
        "token": "Rosa",
        "lemma": "rosa",
        "upos": "NOUN",
    }
    data.update(overrides)
    return NLPAnalysisRecord(**data)


def _fingerprint() -> AnalysisFingerprint:
    policy = DEFAULT_ANALYSIS_EXTRACTION_POLICY
    return AnalysisFingerprint(
        backend="fake",
        language="la",
        processors=policy.processors,
        chunk_size=policy.chunk_chars,
        chunk_strategy=policy.chunk_strategy,
    )


def test_analysis_cache_payload_round_trip(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".analysis_cache"
    text_hash = prepared_text_sha256("Rosa amat.")
    fingerprint = _fingerprint()
    key = build_analysis_cache_key(prepared_text_sha256=text_hash, fingerprint=fingerprint)
    records = [
        _analysis_record(),
        _analysis_record(
            token_index=1,
            global_token_index=1,
            char_start_in_chunk=6,
            char_end_in_chunk=10,
            char_start_in_text=6,
            char_end_in_text=10,
            token="amat",
            lemma=None,
            upos="VERB",
        ),
    ]

    seen, status, payload, meta = get_or_compute_analysis_records(
        cache_dir=cache_dir,
        cache_key=key,
        prepared_text_sha256=text_hash,
        prepared_text_length=len("Rosa amat."),
        fingerprint=fingerprint,
        compute_records=lambda: iter(records),
    )

    assert status == "miss"
    assert list(seen) == records
    assert list(read_analysis_records(payload, meta)) == records

    cached, status, _payload, _meta = get_or_compute_analysis_records(
        cache_dir=cache_dir,
        cache_key=key,
        prepared_text_sha256=text_hash,
        prepared_text_length=len("Rosa amat."),
        fingerprint=fingerprint,
        compute_records=lambda: (_ for _ in ()).throw(AssertionError("should not compute")),
    )
    assert status == "hit"
    assert list(cached) == records


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str) -> NLPDocument:
        self.calls.append(text)
        return NLPDocument(
            sentences=[
                NLPSentence(
                    text="Rosa Marcus xiv a",
                    tokens=[
                        NLPToken("Rosa", "rosa", "NOUN", 0, 4),
                        NLPToken("Marcus", "marcus", "PROPN", 5, 11),
                        NLPToken("xiv", "xiv", "NOUN", 12, 15),
                        NLPToken("a", "a", "NOUN", 16, 17),
                    ],
                )
            ]
        )


def _run(
    tmp_path: Path,
    config_text: str,
    backend: FakeBackend,
    *,
    extraction_policy: AnalysisExtractionPolicy = DEFAULT_ANALYSIS_EXTRACTION_POLICY,
) -> dict[str, int]:
    (tmp_path / "input").mkdir(exist_ok=True)
    (tmp_path / "input" / "text.txt").write_text("Rosa Marcus xiv a", encoding="utf-8")
    config_path = tmp_path / "groups.config.yml"
    config_path.write_text(config_text, encoding="utf-8")
    rc = runner_mod.run(
        project_root=tmp_path,
        config_path=config_path,
        dependencies=runner_dependencies(
            load_config,
            lambda config: BuiltNLPBackend(
                backend=backend,
                info=NLPBackendInfo(name="fake", language=config.language, package="package_a"),
            ),
            extraction_policy=extraction_policy,
        ),
    )
    assert rc.exit_code == 0
    with (tmp_path / "output" / "frequency_text.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["lemma"]: int(row["count"]) for row in rows}


def test_runner_analysis_cache_hit_and_filter_change_reuses_records(tmp_path: Path) -> None:
    backend = FakeBackend()
    base = """
groups:
  text: {files: [input/text.txt]}
analysis_cache:
  enabled: true
  dir: .analysis_cache
artifacts:
  tokens:
    enabled: true
    path: output/tokens.tsv
filters:
  upos_targets: [NOUN]
  min_token_length: 2
  drop_roman_numerals: true
"""
    assert _run(tmp_path, base, backend) == {"rosa": 1}
    assert len(backend.calls) == 1
    first_meta = json.loads((tmp_path / "output" / "run_meta.json").read_text(encoding="utf-8"))
    assert first_meta["analysis_cache"]["misses"] == 1
    assert ".analysis_cache" not in "\n".join(first_meta["generated_outputs"])

    changed_filters = base.replace("min_token_length: 2", "min_token_length: 1").replace(
        "drop_roman_numerals: true",
        "drop_roman_numerals: false",
    ).replace("upos_targets: [NOUN]", "upos_targets: [NOUN, PROPN]")
    assert _run(tmp_path, changed_filters, backend) == {
        "rosa": 1,
        "marcus": 1,
        "xiv": 1,
        "a": 1,
    }
    assert len(backend.calls) == 1
    second_meta = json.loads((tmp_path / "output" / "run_meta.json").read_text(encoding="utf-8"))
    assert second_meta["analysis_cache"]["hits"] == 1
    assert second_meta["analysis_cache"]["misses"] == 0


def test_analysis_cache_policy_change_causes_miss(tmp_path: Path) -> None:
    backend = FakeBackend()
    config = """
groups:
  text: {files: [input/text.txt]}
analysis_cache:
  enabled: true
  dir: .analysis_cache
"""
    first = AnalysisExtractionPolicy(chunk_chars=10)
    second = AnalysisExtractionPolicy(chunk_chars=11)

    _run(tmp_path, config, backend, extraction_policy=first)
    calls_after_first = len(backend.calls)
    _run(tmp_path, config, backend, extraction_policy=first)
    assert len(backend.calls) == calls_after_first
    _run(tmp_path, config, backend, extraction_policy=second)
    assert len(backend.calls) > calls_after_first


def test_corrupted_analysis_cache_is_recomputed(tmp_path: Path) -> None:
    backend = FakeBackend()
    config = """
groups:
  text: {files: [input/text.txt]}
analysis_cache:
  enabled: true
  dir: .analysis_cache
"""
    assert _run(tmp_path, config, backend) == {"rosa": 1, "xiv": 1, "a": 1}
    assert len(backend.calls) == 1

    payloads = list((tmp_path / ".analysis_cache" / "objects").rglob("*.jsonl"))
    assert len(payloads) == 1
    metadata = cache_metadata_path(payloads[0])
    data = json.loads(metadata.read_text(encoding="utf-8"))
    data["payload_sha256"] = "bad"
    metadata.write_text(json.dumps(data), encoding="utf-8")

    assert _run(tmp_path, config, backend) == {"rosa": 1, "xiv": 1, "a": 1}
    assert len(backend.calls) == 2


def test_prune_analysis_cache_removes_payload_metadata_pairs(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".analysis_cache"
    text_hash = prepared_text_sha256("old")
    fingerprint = _fingerprint()
    key = build_analysis_cache_key(prepared_text_sha256=text_hash, fingerprint=fingerprint)
    records, status, payload, meta = get_or_compute_analysis_records(
        cache_dir=cache_dir,
        cache_key=key,
        prepared_text_sha256=text_hash,
        prepared_text_length=3,
        fingerprint=fingerprint,
        compute_records=lambda: iter([_analysis_record()]),
    )
    assert status == "miss"
    assert list(records)
    old_time = 1
    payload.touch()
    meta.touch()
    import os

    os.utime(payload, (old_time, old_time))
    os.utime(meta, (old_time, old_time))

    report = prune_analysis_cache(
        cache_dir,
        keep_days=0,
        keep_objects=0,
        lock_ttl_sec=0,
    )

    assert isinstance(report, cache_storage.PruneReport)
    assert report.removed_objects == 1
    assert not payload.exists()
    assert not meta.exists()


def test_analysis_cache_releases_lock_when_compute_fails(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".analysis_cache"
    text_hash = prepared_text_sha256("boom")
    fingerprint = _fingerprint()
    key = build_analysis_cache_key(prepared_text_sha256=text_hash, fingerprint=fingerprint)
    lock_path = cache_lock_path(cache_dir.resolve(), key)

    def fail():
        raise RuntimeError("boom")
        yield

    records, status, _payload, _meta = get_or_compute_analysis_records(
        cache_dir=cache_dir,
        cache_key=key,
        prepared_text_sha256=text_hash,
        prepared_text_length=4,
        fingerprint=fingerprint,
        compute_records=fail,
    )

    assert status == "miss"
    with pytest.raises(RuntimeError, match="boom"):
        list(records)
    assert not lock_path.exists()
