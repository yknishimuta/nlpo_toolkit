from __future__ import annotations

from dataclasses import replace

import pytest

from nlpo_toolkit.nlp.contracts import NLPBackendInfo
from nlpo_toolkit.corpus_analysis.analysis_cache import build_analysis_cache_key
from nlpo_toolkit.corpus_analysis.analysis_execution import build_analysis_fingerprint
from nlpo_toolkit.corpus_analysis.analysis_policy import (
    AnalysisExtractionPolicy,
    DEFAULT_ANALYSIS_EXTRACTION_POLICY,
)


@pytest.mark.parametrize("chunk_chars", [0, -1, True])
def test_policy_rejects_invalid_chunk_size(chunk_chars: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        AnalysisExtractionPolicy(chunk_chars=chunk_chars)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("processors", "message"),
    [
        ((), "must not be empty"),
        (("tokenize", " "), "empty names"),
        (("tokenize", "tokenize"), "unique"),
    ],
)
def test_policy_rejects_invalid_processors(
    processors: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AnalysisExtractionPolicy(processors=processors)


def test_policy_normalizes_processor_names_without_reordering() -> None:
    policy = AnalysisExtractionPolicy(
        chunk_chars=10,
        processors=(" tokenize ", "pos", "lemma"),
    )
    assert policy.processors == ("tokenize", "pos", "lemma")


def test_policy_builds_flat_fingerprint_and_invalidates_cache_key() -> None:
    backend = NLPBackendInfo(name="fake", language="la", package="pkg")
    default = DEFAULT_ANALYSIS_EXTRACTION_POLICY
    changed_size = replace(default, chunk_chars=100_000)
    changed_processors = replace(default, processors=("tokenize", "pos", "lemma"))

    default_fingerprint = build_analysis_fingerprint(backend_info=backend, policy=default)
    same_fingerprint = build_analysis_fingerprint(backend_info=backend, policy=default)
    size_fingerprint = build_analysis_fingerprint(backend_info=backend, policy=changed_size)
    processor_fingerprint = build_analysis_fingerprint(
        backend_info=backend,
        policy=changed_processors,
    )

    assert default_fingerprint.processors == ("tokenize", "mwt", "pos", "lemma")
    assert default_fingerprint.chunk_size == 200_000
    assert default_fingerprint.chunk_strategy == "char_whitespace"

    def key(fingerprint):
        return build_analysis_cache_key(
            prepared_text_sha256="abc",
            fingerprint=fingerprint,
        )

    assert key(default_fingerprint) == key(same_fingerprint)
    assert key(default_fingerprint) == (
        "8a0de47df8bcb3234e0ce8d4fddf5dd5335d2d4061d01dd6e3cbed840e5ea4df"
    )
    assert key(default_fingerprint) != key(size_fingerprint)
    assert key(default_fingerprint) != key(processor_fingerprint)
