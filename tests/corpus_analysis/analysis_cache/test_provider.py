from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.analysis_cache.provider import provide_analysis_records
from nlpo_toolkit.corpus_analysis.analysis_policy import AnalysisExtractionPolicy
from nlpo_toolkit.corpus_analysis.ports import (
    AnalysisRecordCacheSettings,
    AnalysisRecordRequest,
)
from nlpo_toolkit.nlp.contracts import (
    BuiltNLPBackend,
    NLPBackendInfo,
    NLPDocument,
    NLPSentence,
    NLPToken,
)


class RecordingBackend:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.fail = fail

    def __call__(self, text: str) -> NLPDocument:
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("compute failed")
        return NLPDocument(
            text=text,
            sentences=(
                NLPSentence(
                    text=text,
                    tokens=(NLPToken(text, text.lower(), "NOUN", 0, len(text)),),
                ),
            ),
        )


def _request(
    directory: Path,
    backend: RecordingBackend,
    *,
    enabled: bool,
    policy: AnalysisExtractionPolicy | None = None,
    info: NLPBackendInfo | None = None,
) -> AnalysisRecordRequest:
    return AnalysisRecordRequest(
        text="Rosa",
        backend=BuiltNLPBackend(
            backend=backend,
            info=info or NLPBackendInfo(name="fake", language="la", package="model"),
        ),
        extraction_policy=policy or AnalysisExtractionPolicy(chunk_chars=100),
        cache=AnalysisRecordCacheSettings(enabled, directory, 1.0),
    )


def test_disabled_provider_computes_records_and_returns_stable_key(tmp_path: Path) -> None:
    backend = RecordingBackend()
    request = _request(tmp_path, backend, enabled=False)
    with provide_analysis_records(request) as source:
        records = list(source.records)
        assert source.cache_status == "disabled"
        assert source.cache_key == (
            "891e48ac579a03b6803671df700cc48455114607963376e9671c8cd14fe51c35"
        )
    assert [record.token for record in records] == ["Rosa"]
    assert backend.calls == ["Rosa"]


def test_provider_miss_then_hit_does_not_recompute(tmp_path: Path) -> None:
    backend = RecordingBackend()
    request = _request(tmp_path, backend, enabled=True)
    with provide_analysis_records(request) as source:
        assert source.cache_status == "miss"
        first = list(source.records)
    with provide_analysis_records(request) as source:
        assert source.cache_status == "hit"
        assert list(source.records) == first
    assert backend.calls == ["Rosa"]


def test_policy_and_backend_metadata_change_provider_key(tmp_path: Path) -> None:
    backend = RecordingBackend()
    base = _request(tmp_path, backend, enabled=False)

    def key(request: AnalysisRecordRequest) -> str:
        with provide_analysis_records(request) as source:
            return source.cache_key

    base_key = key(base)
    assert key(replace(base, extraction_policy=AnalysisExtractionPolicy(chunk_chars=101))) != base_key
    assert key(
        replace(base, extraction_policy=AnalysisExtractionPolicy(processors=("tokenize",)))
    ) != base_key
    assert key(
        replace(
            base,
            backend=BuiltNLPBackend(
                backend=backend,
                info=NLPBackendInfo(name="fake", language="la", package="other"),
            ),
        )
    ) != base_key


def test_provider_early_exit_and_failure_leave_no_complete_cache(tmp_path: Path) -> None:
    backend = RecordingBackend()
    request = _request(tmp_path, backend, enabled=True)
    with provide_analysis_records(request) as source:
        next(source.records)
    assert not tuple(tmp_path.rglob("*.meta.json"))
    assert not tuple(tmp_path.rglob("*.lock"))

    failing = _request(tmp_path, RecordingBackend(fail=True), enabled=True)
    with pytest.raises(RuntimeError, match="compute failed"):
        with provide_analysis_records(failing) as source:
            list(source.records)
    assert not tuple(tmp_path.rglob("*.meta.json"))
    assert not tuple(tmp_path.rglob("*.lock"))
    assert not tuple(tmp_path.rglob("*.tmp"))
