import re
from pathlib import Path
from types import MappingProxyType

from nlpo_toolkit.latin.cleaners.models import CleanerProfile, LineRemoveRule, RuleSet, SubstituteRule
from nlpo_toolkit.latin.cleaners.pipeline import clean_document


def test_pipeline_orders_profile_rules_normalization_and_lexicon() -> None:
    profile = CleanerProfile("scholastic_text", Path("rules"), lambda text: tuple(text.splitlines()[1:]), lambda line: line)
    rules = RuleSet(
        (LineRemoveRule(re.compile("DROP"), name="drop"),),
        (SubstituteRule(re.compile("foo"), "bar", name="sub"),),
    )
    result = clean_document("header\nDROP\nfoo  foo\n\n\n\n", profile=profile, rules=rules, lexicon_map=MappingProxyType({"bar": "baz"}))
    assert result.text == "baz baz\n"
    assert result.events[0].text_snippet == "DROP"
    assert result.events[1].text_snippet == "foo  foo"


def test_pipeline_empty_input_returns_one_newline() -> None:
    profile = CleanerProfile("scholastic_text", Path("rules"), lambda text: tuple(text.splitlines()), lambda line: line)
    assert clean_document("", profile=profile, rules=RuleSet(), lexicon_map=MappingProxyType({})).text == "\n"
