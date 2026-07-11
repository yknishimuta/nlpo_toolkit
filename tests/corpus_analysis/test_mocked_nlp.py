from __future__ import annotations

from collections import Counter

from nlpo_toolkit.models import NLPDocument, NLPSentence, NLPToken
from nlpo_toolkit.nlp import count_nouns


class DummyNLP:
    def __init__(self, doc: NLPDocument):
        self._doc = doc
        self.calls: list[str] = []

    def __call__(self, text: str) -> NLPDocument:
        self.calls.append(text)
        return self._doc


def _nlp(tokens: list[NLPToken]) -> DummyNLP:
    return DummyNLP(NLPDocument(sentences=[NLPSentence(tokens=tokens, text="ignored")]))


def test_count_nouns_uses_analysis_record_pipeline_for_lemma_mode():
    nlp = _nlp(
        [
            NLPToken(text="Puella", lemma="Puella", upos="NOUN", start_char=0, end_char=6),
            NLPToken(text="amat", lemma="amo", upos="VERB", start_char=7, end_char=11),
            NLPToken(text="rosam", lemma="Rosa", upos="NOUN", start_char=12, end_char=17),
            NLPToken(text="LIBER", lemma=None, upos="NOUN", start_char=18, end_char=23),
        ]
    )

    out = count_nouns("whatever", nlp, use_lemma=True, upos_targets={"NOUN"})

    assert out == Counter({"puella": 1, "rosa": 1, "liber": 1})
    assert nlp.calls == ["whatever"]


def test_count_nouns_uses_surface_when_use_lemma_false():
    nlp = _nlp(
        [
            NLPToken(text="Puella", lemma="puella", upos="NOUN", start_char=0, end_char=6),
            NLPToken(text="rosam", lemma="rosa", upos="NOUN", start_char=7, end_char=12),
        ]
    )

    out = count_nouns("whatever", nlp, use_lemma=False, upos_targets={"NOUN"})

    assert out == Counter({"puella": 1, "rosam": 1})


def test_count_nouns_applies_min_length_and_roman_filter(tmp_path):
    exc_file = tmp_path / "exceptions.txt"
    exc_file.write_text("xiv\n", encoding="utf-8")
    nlp = _nlp(
        [
            NLPToken(text="ro", lemma="ro", upos="NOUN", start_char=0, end_char=2),
            NLPToken(text="xiv", lemma="xiv", upos="NOUN", start_char=3, end_char=6),
            NLPToken(text="iv", lemma="iv", upos="NOUN", start_char=7, end_char=9),
            NLPToken(text="rosa", lemma="rosa", upos="NOUN", start_char=10, end_char=14),
        ]
    )

    out = count_nouns(
        "whatever",
        nlp,
        use_lemma=False,
        upos_targets={"NOUN"},
        min_token_length=3,
        drop_roman_numerals=True,
        roman_exceptions_file=exc_file,
    )

    assert out == Counter({"xiv": 1, "rosa": 1})
