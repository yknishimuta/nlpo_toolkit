from pathlib import Path

from .reader import read_token_artifact_metadata, read_token_records
from .schema import TokenArtifactMetadata


def validate_token_artifact(
    artifact_path: Path, *, verify_hash: bool = True
) -> TokenArtifactMetadata:
    metadata = read_token_artifact_metadata(artifact_path)
    for _record in read_token_records(
        artifact_path, require_complete=True, verify_hash=verify_hash
    ):
        pass
    return metadata
