import re

from nlpo_toolkit.latin.cleaners.models import LineRemoveRule, RuleReference, RuleSet, SubstituteRule
from nlpo_toolkit.latin.cleaners.rule_engine import apply_rule_set


def test_rule_engine_removes_then_substitutes_in_order_and_emits_events() -> None:
    lines = ("  DROP this  ", "foo foo")
    rules = RuleSet(
        remove_lines=(LineRemoveRule(re.compile(r"^DROP"), RuleReference(key="drop"), "drop"),),
        substitutions=(
            SubstituteRule(re.compile(r"(foo)"), "bar", RuleReference(key="one"), "first"),
            SubstituteRule(re.compile(r"bar"), "baz", RuleReference(key="two"), "second"),
        ),
    )
    result = apply_rule_set(lines, rules=rules, kind="scholastic_text", doc_id="doc", snippet_chars=4, finalize_line=str.upper)
    assert result.lines == ("BAZ BAZ",)
    assert [(event.action, event.rule_name, event.line_number, event.match_count) for event in result.events] == [
        ("drop_line", "drop", 1, 1),
        ("substitute", "first", 2, 2),
        ("substitute", "second", 2, 2),
    ]
    assert result.events[1].text_snippet == "foo "
    assert lines == ("  DROP this  ", "foo foo")
