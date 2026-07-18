from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pytest

from nlpo_toolkit.corpus_analysis.analysis_cache.codec import (
    decode_record,
    encode_record,
    parse_analysis_record_payload,
)
from nlpo_toolkit.corpus_analysis.analysis_cache.errors import AnalysisCacheError
from .conftest import record
from nlpo_toolkit.nlp.contracts import UDMorphFeature


def test_record_codec_round_trip_with_optional_values(tmp_path: Path) -> None:
    original = record()
    original = replace(original, morphology=(UDMorphFeature("Case", "Nom"),))
    assert (
        decode_record(encode_record(original), path=tmp_path / "payload", line_number=1)
        == original
    )


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
        payload = parse_analysis_record_payload(
            data,
            path=tmp_path / "payload",
            line_number=1,
        )
        decode_record(payload, path=tmp_path / "payload", line_number=1)


def test_cache_rejects_invalid_morphology_payload(tmp_path: Path) -> None:
    data = encode_record(record())
    data["morphology"] = [{"attribute": "Case", "value": "Nom", "extra": "x"}]
    with pytest.raises(AnalysisCacheError, match="morphology"):
        parse_analysis_record_payload(data, path=tmp_path / "payload", line_number=1)
