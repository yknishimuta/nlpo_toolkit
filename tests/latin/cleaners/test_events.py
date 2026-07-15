from pathlib import Path

from nlpo_toolkit.latin.cleaners.events import append_ref_events
from nlpo_toolkit.latin.cleaners.models import RefEvent, RuleReference


def test_events_writer_preserves_schema_order_and_appends(tmp_path: Path) -> None:
    path = tmp_path / "nested/events.tsv"
    event = RefEvent("d", "corpus_corporum", "r", "substitute", 2, 3, RuleReference("k", "a", "w", "l"), "text")
    append_ref_events(path, (event,))
    append_ref_events(path, (event,))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == ["doc_id", "kind", "rule_name", "action", "line_no", "match_count", "ref_key", "ref_author", "ref_work", "ref_loc", "text_snippet"]
    assert len(lines) == 3
