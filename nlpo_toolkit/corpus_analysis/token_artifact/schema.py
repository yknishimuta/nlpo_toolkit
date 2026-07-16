from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from .errors import TokenArtifactMetadataError


TOKEN_ARTIFACT_SCHEMA_NAME = "nlpo-token-artifact"
TOKEN_ARTIFACT_SCHEMA_VERSION = 1
TOKEN_ARTIFACT_FORMAT = "tsv"
TOKEN_ARTIFACT_ENCODING = "utf-8"
TOKEN_ARTIFACT_DELIMITER = "\t"
TOKEN_ARTIFACT_COLUMNS = (
    "group", "source_file", "section", "chunk_index", "sentence_index",
    "token_index", "global_token_index", "char_start_in_chunk",
    "char_end_in_chunk", "char_start_in_text", "char_end_in_text", "sentence",
    "token", "lemma", "upos", "analysis_key", "included", "exclusion_reason",
    "ref_tag",
)


def _json_object(value: Mapping[str, JsonValue], *, field: str) -> MappingProxyType:
    try:
        copied = json.loads(json.dumps(dict(value), ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain only JSON values") from exc
    if not isinstance(copied, dict):
        raise ValueError(f"{field} must be a JSON object")
    return MappingProxyType(copied)


@dataclass(frozen=True)
class TokenArtifactDescriptor:
    group: str
    source_files: tuple[str, ...] = ()
    analysis_unit: str = ""
    upos_targets: tuple[str, ...] = ()
    nlp: Mapping[str, JsonValue] = MappingProxyType({})
    filters: Mapping[str, JsonValue] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.group, str) or not isinstance(self.analysis_unit, str):
            raise TypeError("descriptor group and analysis_unit must be strings")
        source_files = tuple(self.source_files)
        upos_targets = tuple(self.upos_targets)
        if not all(isinstance(item, str) for item in source_files):
            raise TypeError("descriptor source_files must contain strings")
        if not all(isinstance(item, str) for item in upos_targets):
            raise TypeError("descriptor upos_targets must contain strings")
        object.__setattr__(self, "source_files", source_files)
        object.__setattr__(self, "upos_targets", upos_targets)
        object.__setattr__(self, "nlp", _json_object(self.nlp, field="nlp"))
        object.__setattr__(self, "filters", _json_object(self.filters, field="filters"))


class TokenArtifactMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    schema_name: Literal["nlpo-token-artifact"] = Field(
        default=TOKEN_ARTIFACT_SCHEMA_NAME, alias="schema"
    )
    schema_version: Literal[1] = TOKEN_ARTIFACT_SCHEMA_VERSION
    format: Literal["tsv"] = TOKEN_ARTIFACT_FORMAT
    encoding: Literal["utf-8"] = TOKEN_ARTIFACT_ENCODING
    delimiter: Literal["\t"] = TOKEN_ARTIFACT_DELIMITER
    complete: StrictBool
    row_count: StrictInt
    included_row_count: StrictInt
    excluded_row_count: StrictInt
    group: StrictStr
    source_files: tuple[StrictStr, ...]
    analysis_unit: StrictStr
    upos_targets: tuple[StrictStr, ...]
    nlp: dict[StrictStr, JsonValue]
    filters: dict[StrictStr, JsonValue]
    artifact_path: StrictStr
    sha256: StrictStr
    size_bytes: StrictInt

    @property
    def schema(self) -> str:
        return self.schema_name

    @field_validator(
        "row_count", "included_row_count", "excluded_row_count", "size_bytes"
    )
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be non-negative")
        return value

    @model_validator(mode="after")
    def _invariants(self) -> "TokenArtifactMetadata":
        if self.included_row_count + self.excluded_row_count != self.row_count:
            raise ValueError("included_row_count + excluded_row_count must equal row_count")
        if self.complete and (
            len(self.sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.sha256)
        ):
            raise ValueError("complete artifact sha256 must be 64 lowercase hex characters")
        return self


def metadata_to_json(metadata: TokenArtifactMetadata) -> str:
    data = metadata.model_dump(mode="json", by_alias=True)
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def metadata_from_json(text: str, *, source_path: Path) -> TokenArtifactMetadata:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TokenArtifactMetadataError(
            f"Invalid token artifact metadata JSON: {source_path.resolve()}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise TokenArtifactMetadataError(
            f"Token artifact metadata root must be an object: {source_path.resolve()}"
        )
    if raw.get("schema") != TOKEN_ARTIFACT_SCHEMA_NAME:
        raise TokenArtifactMetadataError(
            f"Unsupported token artifact schema {raw.get('schema')!r}: "
            f"{source_path.resolve()}"
        )
    if raw.get("schema_version") != TOKEN_ARTIFACT_SCHEMA_VERSION:
        raise TokenArtifactMetadataError(
            f"Unsupported token artifact schema version "
            f"{raw.get('schema_version')!r}: {source_path.resolve()}"
        )
    normalized = dict(raw)
    for field in ("source_files", "upos_targets"):
        value = normalized.get(field)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise TokenArtifactMetadataError(
                f"Token artifact metadata {field} must be an array of strings: "
                f"{source_path.resolve()}"
            )
        normalized[field] = tuple(value)
    for field in ("nlp", "filters"):
        if not isinstance(normalized.get(field), dict):
            raise TokenArtifactMetadataError(
                f"Token artifact metadata {field} must be an object: "
                f"{source_path.resolve()}"
            )
    try:
        return TokenArtifactMetadata.model_validate(normalized, strict=True)
    except ValidationError as exc:
        raise TokenArtifactMetadataError(
            f"Invalid token artifact metadata schema: {source_path.resolve()}: {exc}"
        ) from exc
