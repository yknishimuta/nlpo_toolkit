from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.analysis_records import (
    TokenRecord,
    counter_from_token_records,
    iter_token_records,
)
from nlpo_toolkit.corpus_analysis.token_artifact import (
    TOKEN_ARTIFACT_COLUMNS,
    TOKEN_ARTIFACT_SCHEMA_NAME,
    TOKEN_ARTIFACT_SCHEMA_VERSION,
    TokenArtifactError,
    TokenArtifactMetadata,
    TokenArtifactWriter,
    read_token_records,
    token_artifact_metadata_path,
    validate_token_artifact,
)
from nlpo_toolkit.nlp.contracts import NLPDocument, NLPSentence, NLPToken


def _record(**overrides) -> TokenRecord:
    data = {
        "group": "text",
        "source_file": "input/text.txt",
        "section": None,
        "chunk_index": 0,
        "sentence_index": 0,
        "token_index": 0,
        "global_token_index": 0,
        "char_start_in_chunk": 0,
        "char_end_in_chunk": 6,
        "char_start_in_text": 0,
        "char_end_in_text": 6,
        "sentence": "Puella amat.",
        "token": "Puella",
        "lemma": "puella",
        "upos": "NOUN",
        "analysis_key": "puella",
        "included": True,
        "exclusion_reason": None,
        "ref_tag": None,
    }
    data.update(overrides)
    return TokenRecord(**data)


def test_token_artifact_schema_constants() -> None:
    assert TOKEN_ARTIFACT_SCHEMA_NAME == "nlpo-token-artifact"
    assert TOKEN_ARTIFACT_SCHEMA_VERSION == 1
    assert TOKEN_ARTIFACT_COLUMNS[0] == "group"
    assert "char_end_in_text" in TOKEN_ARTIFACT_COLUMNS
    assert TOKEN_ARTIFACT_COLUMNS[-1] == "ref_tag"


def test_token_artifact_round_trip_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "tokens.tsv"
    metadata = TokenArtifactMetadata(
        group="text",
        source_files=("input/text.txt",),
        analysis_unit="lemma",
        upos_targets=("NOUN",),
        nlp={"backend": "fake"},
        filters={"min_token_length": 2},
    )
    records = [
        _record(),
        _record(
            token_index=1,
            global_token_index=1,
            token="amat",
            lemma="amo",
            upos="VERB",
            analysis_key="amo",
            included=False,
            exclusion_reason="upos_not_targeted",
        ),
    ]

    with TokenArtifactWriter(path, token_artifact_metadata_path(path), metadata=metadata) as writer:
        for record in records:
            writer.write(record)

    assert list(read_token_records(path)) == records
    meta = validate_token_artifact(path)
    assert meta.row_count == 2
    assert meta.included_row_count == 1
    assert meta.excluded_row_count == 1
    assert meta.complete is True
    assert meta.sha256
    assert meta.size_bytes > 0
    assert token_artifact_metadata_path(path).name == "tokens.meta.json"


def test_counter_from_token_records_counts_only_included_keys() -> None:
    records = [
        _record(analysis_key="puella", included=True),
        _record(analysis_key="amo", included=False, exclusion_reason="upos_not_targeted"),
        _record(analysis_key=None, included=True),
    ]
    assert counter_from_token_records(records) == Counter({"puella": 1})


def test_token_artifact_atomic_write_does_not_publish_failed_files(tmp_path: Path) -> None:
    path = tmp_path / "tokens.tsv"
    with pytest.raises(RuntimeError):
        with TokenArtifactWriter(path, token_artifact_metadata_path(path), metadata=TokenArtifactMetadata(group="text")) as writer:
            writer.write(_record())
            raise RuntimeError("boom")

    assert not path.exists()
    assert not token_artifact_metadata_path(path).exists()
    assert not (tmp_path / "tokens.tsv.tmp").exists()


def test_token_artifact_reader_rejects_row_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "tokens.tsv"
    with TokenArtifactWriter(path, token_artifact_metadata_path(path), metadata=TokenArtifactMetadata(group="text")) as writer:
        writer.write(_record())

    meta_path = token_artifact_metadata_path(path)
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    data["row_count"] = 2
    meta_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(TokenArtifactError, match="row_count mismatch"):
        list(read_token_records(path))


def test_token_artifact_reader_rejects_bad_bool(tmp_path: Path) -> None:
    path = tmp_path / "tokens.tsv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(TOKEN_ARTIFACT_COLUMNS)
        row = [
            "text",
            "",
            "",
            "0",
            "0",
            "0",
            "0",
            "",
            "",
            "",
            "",
            "",
            "arma",
            "arma",
            "NOUN",
            "arma",
            "yes",
            "",
            "",
        ]
        writer.writerow(row)
    token_artifact_metadata_path(path).write_text(
        json.dumps(TokenArtifactMetadata(group="text", row_count=1).to_dict()),
        encoding="utf-8",
    )

    with pytest.raises(TokenArtifactError, match="Invalid boolean"):
        list(read_token_records(path))


class _FakeBackend:
    def __call__(self, text: str) -> NLPDocument:
        return NLPDocument(
            sentences=[
                NLPSentence(
                    text="amat a xiv",
                    tokens=[
                        NLPToken("amat", "amo", "VERB", 0, 4),
                        NLPToken("a", "a", "NOUN", 5, 6),
                        NLPToken("xiv", "xiv", "NOUN", 7, 10),
                        NLPToken("", None, "NOUN", 11, 11),
                    ],
                )
            ]
        )


def test_iter_token_records_exclusion_reasons(tmp_path: Path) -> None:
    records = list(
        iter_token_records(
            text="amat a xiv",
            nlp=_FakeBackend(),
            group="text",
            source_files=(tmp_path / "input.txt",),
            use_lemma=True,
            upos_targets={"NOUN"},
            min_token_length=2,
            drop_roman_numerals=True,
        )
    )

    assert [record.exclusion_reason for record in records] == [
        "upos_not_targeted",
        "too_short",
        "roman_numeral",
        "missing_key",
    ]
