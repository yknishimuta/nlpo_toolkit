from nlpo_toolkit.latin.cleaners.corpora import corpus_corporum, scholastic


def test_corpus_corporum_profile_owns_header_and_tab_behavior() -> None:
    assert corpus_corporum.prepare_lines("meta\n  #####  \nbody\n######\nlater") == ("body", "######", "later")
    assert corpus_corporum.prepare_lines("####\nbody") == ("####", "body")
    assert corpus_corporum.finalize_line("a\tb") == "a b"
    assert corpus_corporum.PROFILE.kind == "corpus_corporum"
    assert corpus_corporum.DEFAULT_RULES_PATH.name == "corpus_corporum.yml"


def test_scholastic_profile_keeps_headers_and_tabs() -> None:
    assert scholastic.prepare_lines("#####\na\tb") == ("#####", "a\tb")
    assert scholastic.finalize_line("a\tb") == "a\tb"
    assert scholastic.PROFILE.kind == "scholastic_text"
    assert scholastic.DEFAULT_RULES_PATH.name == "scholastic_text.yml"
