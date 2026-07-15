from __future__ import annotations

from collections.abc import Callable

from nlpo_toolkit.cleaner_contracts import CleanerKind

from .models import RefEvent, RuleApplicationResult, RuleSet


def _identity(line: str) -> str:
    return line


def apply_rule_set(
    lines: tuple[str, ...],
    *,
    rules: RuleSet,
    kind: CleanerKind,
    doc_id: str = "",
    snippet_chars: int = 200,
    finalize_line: Callable[[str], str] = _identity,
) -> RuleApplicationResult:
    output: list[str] = []
    events: list[RefEvent] = []
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        dropped = False
        for rule in rules.remove_lines:
            if rule.pattern.match(stripped):
                events.append(RefEvent(doc_id, kind, rule.name, "drop_line", line_number, 1, rule.reference, line[:snippet_chars]))
                dropped = True
                break
        if dropped:
            continue
        for rule in rules.substitutions:
            match_count = sum(1 for _match in rule.pattern.finditer(line))
            if match_count:
                snippet = line[:snippet_chars]
                line = rule.pattern.sub(rule.replacement, line)
                events.append(RefEvent(doc_id, kind, rule.name, "substitute", line_number, match_count, rule.reference, snippet))
        output.append(finalize_line(line))
    return RuleApplicationResult(tuple(output), tuple(events))
