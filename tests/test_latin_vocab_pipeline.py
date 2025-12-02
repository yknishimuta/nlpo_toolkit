from __future__ import annotations

from pathlib import Path
from collections import Counter
import json

import pytest

from nlpo_toolkit.nlp import (
    build_stanza_pipeline,
    tokenize_all_pos,
    count_nouns,
    count_nouns_streaming,
    normalize_token,
)

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "input"
EXPECTED_DIR = HERE / "expected_output"


# =========================================================
# fixtures
# =========================================================

@pytest.fixture(scope="session")
def test_text_1() -> str:
    path = INPUT_DIR / "test_input_1.txt"
    assert path.is_file(), f"missing: {path}"
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def test_text_2() -> str:
    path = INPUT_DIR / "test_input_2.txt"
    assert path.is_file(), f"missing: {path}"
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def expected_nouns_test1() -> dict:
    path = EXPECTED_DIR / "nouns_test1.json"
    assert path.is_file(), f"missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def expected_normalize_test2() -> dict:
    path = EXPECTED_DIR / "normalize_test2.json"
    assert path.is_file(), f"missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def nlp():
    """
    Stanza Latin pipeline (fixed to 'perseus').
    Skip tests if stanza is not available.
    """
    pytest.importorskip("stanza")
    pipeline = build_stanza_pipeline(
        lang="la",
        processors="tokenize,mwt,pos,lemma",
        package="perseus",
        use_gpu=False,
    )
    return pipeline


@pytest.fixture(scope="session")
def nlp_utils():
    """A thin wrapper providing NLP utility functions."""
    return {
        "tokenize_all_pos": tokenize_all_pos,
        "count_nouns": count_nouns,
        "count_nouns_streaming": count_nouns_streaming,
        "normalize_token": normalize_token,
    }


# =========================================================
# tests
# =========================================================

def test_tokenize_all_pos_basic(test_text_1: str, nlp_utils):
    """Verify that tokenize_all_pos splits tokens as expected."""
    tokenize_all_pos = nlp_utils["tokenize_all_pos"]

    tokens = tokenize_all_pos(test_text_1)
    expected = [
        "puella",
        "rosam",
        "amat",
        "rosa",
        "pulchra",
        "est",
        "puella",
        "librum",
        "legit",
    ]
    assert tokens == expected


def test_normalize_token_ligatures_and_diacritics(nlp_utils):
    """
    Verify the behavior of normalize_token:
    - expands ligatures
    - strips diacritics
    - lowercases the token
    """
    normalize_token = nlp_utils["normalize_token"]

    assert normalize_token("Cæsar") == "caesar"
    assert normalize_token("multās") == "multas"
    assert normalize_token("victōriās") == "victorias"
    assert normalize_token("Œconomia") == "oeconomia"


def test_count_nouns_lemma_matches_expected(
    test_text_1,
    expected_nouns_test1,
    nlp,
    nlp_utils,
):
    """Check that lemma-based noun counts match expected values."""
    count_nouns = nlp_utils["count_nouns"]

    expected_lemma_counts = expected_nouns_test1["lemma_counts"]
    counter = count_nouns(test_text_1, nlp, use_lemma=True, upos_targets={"NOUN"})

    assert counter == Counter(expected_lemma_counts)


def test_count_nouns_surface_matches_expected(
    test_text_1,
    expected_nouns_test1,
    nlp,
    nlp_utils,
):
    """
    Check that surface-form noun counts match expected values.

    The implementation lowercases tokens,
    so expected values are compared after lowercasing keys.
    """
    count_nouns = nlp_utils["count_nouns"]

    raw_expected = expected_nouns_test1["surface_counts"]
    expected_surface_counts = {k.lower(): v for k, v in raw_expected.items()}

    counter = count_nouns(test_text_1, nlp, use_lemma=False, upos_targets={"NOUN"})
    assert counter == Counter(expected_surface_counts)


def test_count_nouns_streaming_equals_non_streaming(
    test_text_1,
    nlp,
    nlp_utils,
):
    """
    Verify that the streaming version (count_nouns_streaming)
    produces the same result as the non-streaming version.
    """
    count_nouns = nlp_utils["count_nouns"]
    count_nouns_streaming = nlp_utils["count_nouns_streaming"]

    baseline = count_nouns(test_text_1, nlp, use_lemma=True, upos_targets={"NOUN"})
    streaming = count_nouns_streaming(
        test_text_1,
        nlp,
        use_lemma=True,
        upos_targets={"NOUN"},
        chunk_chars=20,
        label="test_chunk",
    )

    assert streaming == baseline


def test_normalize_on_test_input_2(
    test_text_2,
    expected_normalize_test2,
    nlp_utils,
):
    """
    For test_input_2, verify that all expected normalized tokens
    appear in the multiset of actual normalized tokens.
    """
    tokenize_all_pos = nlp_utils["tokenize_all_pos"]
    normalize_token = nlp_utils["normalize_token"]

    raw_tokens = tokenize_all_pos(test_text_2)
    normalized_tokens = [normalize_token(t) for t in raw_tokens]

    expected = expected_normalize_test2["normalized_tokens"]
    for tok in expected:
        assert tok in normalized_tokens
