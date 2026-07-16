from __future__ import annotations

from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.analysis_cache.codec import decode_record, encode_record
from nlpo_toolkit.corpus_analysis.analysis_cache.errors import AnalysisCacheError
from .conftest import record


def test_record_codec_round_trip_with_optional_values(tmp_path: Path) -> None:
    original = record()
    assert decode_record(
        encode_record(original), path=tmp_path / "payload", line_number=1
    ) == original


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"chunk_index": True}, "integer"),
        ({"token": 3}, "token analysis"),
        ({"sentence": []}, "token analysis"),
        ({"lemma": 1}, "lemma"),
        ({"upos": {}}, "upos"),
    ],
)
def test_decode_rejects_invalid_fields(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    data = encode_record(record())
    data.update(change)
    with pytest.raises(AnalysisCacheError, match=message):
        decode_record(data, path=tmp_path / "payload", line_number=1)
