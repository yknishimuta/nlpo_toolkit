from nlpo_toolkit.latin.latin_wordlist.engine import (
    iter_latin_word_candidates,
    merge_wordlist_candidates,
    select_frequent_forms,
)
from nlpo_toolkit.latin.latin_wordlist.models import (
    ConlluCandidates,
    ExtraWordlistCandidates,
    TextCandidates,
    WordlistFilterPolicy,
    WordlistTokenizationPolicy,
)
from pathlib import Path


def test_tokenization_preserves_existing_semantics() -> None:
    words = tuple(
        iter_latin_word_candidates(
            "Rōsa—AMAT! x 12 vir_que",
            policy=WordlistTokenizationPolicy(extra_punct="—"),
            min_length=2,
        )
    )
    assert words == ("rōsa", "amat", "vir", "que")


def test_threshold_and_merge_are_deterministic_without_mutating_inputs() -> None:
    counts = {"rosa": 1, "amo": 2}
    conllu = ConlluCandidates((), frozenset({"lemma"}), counts)
    text = TextCandidates((), {"textus": 3, "rarus": 1})
    extra_entries = {"zeta", "amo"}
    extra = ExtraWordlistCandidates(Path("extra"), extra_entries)
    result = merge_wordlist_candidates(
        conllu=conllu,
        text=text,
        extras=(extra,),
        filters=WordlistFilterPolicy(2, 2, 3),
    )
    assert result == ("amo", "lemma", "textus", "zeta")
    assert select_frequent_forms(counts, minimum_frequency=2) == {"amo"}
    assert counts == {"rosa": 1, "amo": 2}
    assert extra_entries == {"zeta", "amo"}
