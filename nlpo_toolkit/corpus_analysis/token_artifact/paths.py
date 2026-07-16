from pathlib import Path


def token_artifact_metadata_path(tsv_path: Path) -> Path:
    path = Path(tsv_path)
    return path.with_name(f"{path.stem}.meta.json")
