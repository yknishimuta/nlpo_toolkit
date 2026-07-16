from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from .models import RefEvent


_COLUMNS = ("doc_id", "kind", "rule_name", "action", "line_no", "match_count", "ref_key", "ref_author", "ref_work", "ref_loc", "text_snippet")


def write_ref_events(path: str | Path, events: Sequence[RefEvent]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(_COLUMNS)
        for event in events:
            reference = event.reference
            writer.writerow((event.doc_id, event.kind, event.rule_name, event.action, event.line_number, event.match_count, reference.key, reference.author, reference.work, reference.location, event.text_snippet))
    temporary.replace(target)
