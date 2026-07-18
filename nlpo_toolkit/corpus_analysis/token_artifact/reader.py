from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from ..analysis_records import TokenRecord
from .codec import decode_token_record
from .errors import (
    TokenArtifactFormatError,
    TokenArtifactIntegrityError,
    TokenArtifactMetadataError,
)
from .integrity import file_size, sha256_file
from .paths import token_artifact_metadata_path
from .schema import (
    TOKEN_ARTIFACT_V1_COLUMNS,
    TOKEN_ARTIFACT_V2_COLUMNS,
    TOKEN_ARTIFACT_DELIMITER,
    TOKEN_ARTIFACT_ENCODING,
    TokenArtifactMetadata,
    metadata_from_json,
)


def read_token_artifact_metadata(artifact_path: Path) -> TokenArtifactMetadata:
    artifact = Path(artifact_path).resolve()
    path = token_artifact_metadata_path(artifact)
    if not path.exists():
        raise TokenArtifactMetadataError(
            f"Token artifact metadata was not found: {path}"
        )
    if not path.is_file():
        raise TokenArtifactMetadataError(
            f"Token artifact metadata is not a regular file: {path}"
        )
    try:
        text = path.read_text(encoding=TOKEN_ARTIFACT_ENCODING)
    except UnicodeError as exc:
        raise TokenArtifactMetadataError(
            f"Token artifact metadata is not valid UTF-8: {path}"
        ) from exc
    except OSError as exc:
        raise TokenArtifactMetadataError(
            f"Failed to read token artifact metadata: {path}: {exc}"
        ) from exc
    return metadata_from_json(text, source_path=path)


def read_token_records(
    artifact_path: Path,
    *,
    require_complete: bool = True,
    verify_hash: bool = False,
) -> Iterator[TokenRecord]:
    """Read records; integrity counts are checked only if iteration completes."""
    path = Path(artifact_path).resolve()
    if not path.exists():
        raise TokenArtifactFormatError(f"Token artifact not found: {path}")
    if not path.is_file():
        raise TokenArtifactFormatError(f"Token artifact is not a regular file: {path}")
    metadata = read_token_artifact_metadata(path)
    if require_complete and not metadata.complete:
        raise TokenArtifactIntegrityError(f"Token artifact is incomplete: {path}")

    row_count = included_count = excluded_count = 0
    try:
        stream = path.open("r", encoding=TOKEN_ARTIFACT_ENCODING, newline="")
    except UnicodeError as exc:
        raise TokenArtifactFormatError(
            f"Token artifact is not valid UTF-8: {path}"
        ) from exc
    except OSError as exc:
        raise TokenArtifactFormatError(
            f"Failed to read token artifact: {path}: {exc}"
        ) from exc
    try:
        reader = csv.DictReader(stream, delimiter=TOKEN_ARTIFACT_DELIMITER)
        header = tuple(reader.fieldnames or ())
        expected_columns = (
            TOKEN_ARTIFACT_V1_COLUMNS
            if metadata.schema_version == 1
            else TOKEN_ARTIFACT_V2_COLUMNS
        )
        if header != expected_columns:
            raise TokenArtifactFormatError(
                f"Token artifact header does not match schema: {path}; "
                f"expected={expected_columns}; actual={header}"
            )
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise TokenArtifactFormatError(
                    f"Malformed token artifact row at {path}:{line_number}"
                )
            record = decode_token_record(
                row,
                source_path=path,
                line_number=line_number,
                schema_version=metadata.schema_version,
            )
            row_count += 1
            if record.included:
                included_count += 1
            else:
                excluded_count += 1
            yield record
    except UnicodeError as exc:
        raise TokenArtifactFormatError(
            f"Token artifact is not valid UTF-8: {path}"
        ) from exc
    finally:
        stream.close()

    observed = (row_count, included_count, excluded_count)
    expected = (
        metadata.row_count,
        metadata.included_row_count,
        metadata.excluded_row_count,
    )
    if observed != expected:
        raise TokenArtifactIntegrityError(
            f"Token artifact count mismatch: {path}; expected={expected}; actual={observed}"
        )
    if verify_hash:
        actual_size = file_size(path)
        if actual_size != metadata.size_bytes:
            raise TokenArtifactIntegrityError(
                f"Token artifact size mismatch: {path}; "
                f"expected={metadata.size_bytes}; actual={actual_size}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != metadata.sha256:
            raise TokenArtifactIntegrityError(
                f"Token artifact hash mismatch: {path}; "
                f"expected={metadata.sha256}; actual={actual_hash}"
            )
