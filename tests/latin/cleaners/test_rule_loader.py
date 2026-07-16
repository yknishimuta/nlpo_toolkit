from pathlib import Path

import pytest

from nlpo_toolkit.latin.cleaners.errors import CleanerRuleConfigError
from nlpo_toolkit.latin.cleaners.rule_loader import load_rule_set


def test_load_rule_set_types_references_order_and_disabled(tmp_path: Path) -> None:
    path = tmp_path / "rules.yml"
    path.write_text(
        "remove_line_patterns:\n"
        "  - {name: first, pattern: '^DROP', ref: 'Aristotle:Metaphys'}\n"
        "  - {name: off, enabled: false, pattern: '^OFF'}\n"
        "substitute_patterns:\n"
        "  - name: replace\n    pattern: 'foo'\n    repl: bar\n"
        "    ref: {author: Thomas, work: Summa, loc: I.1}\n",
        encoding="utf-8",
    )
    rules = load_rule_set(path)
    assert tuple(rule.name for rule in rules.remove_lines) == ("first",)
    assert rules.remove_lines[0].reference.author == "Aristotle"
    assert rules.substitutions[0].replacement == "bar"
    assert rules.substitutions[0].reference.key == "Thomas:Summa"


@pytest.mark.parametrize(
    "text, message",
    [
        ("- invalid\n", "Top-level YAML"),
        ("remove_line_patterns: {}\n", "must be a list"),
        ("remove_line_patterns: [x]\n", "must be a mapping"),
        ("remove_line_patterns: [{pattern: ''}]\n", "non-empty"),
        ("remove_line_patterns: [{pattern: '[']} ]\n", "Invalid YAML"),
        ("remove_line_patterns: [{pattern: '[', enabled: true}]\n", "invalid regex"),
        ("substitute_patterns: [{pattern: x, repl: 1}]\n", "repl must be a string"),
        ("remove_line_patterns: [{pattern: x, ref: 1}]\n", "ref must be"),
    ],
)
def test_load_rule_set_rejects_invalid_schema(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "rules.yml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(CleanerRuleConfigError, match=message):
        load_rule_set(path)


@pytest.mark.parametrize("text", (
    "remove_line_patterns: []\nremove_line_patterns: []\n",
    "remove_line_patterns:\n  - pattern: x\n    pattern: y\n",
))
def test_rule_yaml_rejects_duplicate_sections_and_rule_fields(
    tmp_path: Path, text: str
) -> None:
    path = tmp_path / "rules.yml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(CleanerRuleConfigError, match="Duplicate YAML key"):
        load_rule_set(path)
