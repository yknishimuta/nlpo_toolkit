from collections import Counter
from nlpo_toolkit.corpus_analysis.counters import filter_counter

def test_filter_counter_excludes_words():
    c = Counter({"idest": 3, "rosa": 2})
    out = filter_counter(c, exclude={"idest"})
    assert "idest" not in out
    assert out["rosa"] == 2
