from __future__ import annotations

import os
import pytest

from nlpo_toolkit.nlp import build_stanza_pipeline, count_nouns


@pytest.mark.stanza_integration
@pytest.fixture(scope="session")
def nlp():
    if os.environ.get("RUN_STANZA_TESTS") != "1":
        pytest.skip("Set RUN_STANZA_TESTS=1 to run stanza integration tests")

    pytest.importorskip("stanza")

    return build_stanza_pipeline(
        lang="la",
        processors="tokenize,mwt,pos,lemma",
        package="perseus",
        use_gpu=False,
    )


@pytest.mark.stanza_integration
def test_count_nouns_with_real_stanza(nlp):
    text = "Puella rosam amat. Puella legit."
    out = count_nouns(text, nlp, use_lemma=True, upos_targets={"NOUN"})
    assert out["puella"] >= 1

