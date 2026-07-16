from __future__ import annotations

import csv
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator, Sequence

from .models import PlannedArtifact


class ArtifactPublicationError(RuntimeError):
    """Raised when a planned artifact cannot be published."""


def _owner(artifact: PlannedArtifact) -> str:
    if artifact.group is not None:
        return f"group={artifact.group}"
    if artifact.name is not None:
        return f"name={artifact.name}"
    return "run"


def _error(artifact: PlannedArtifact, stage: str, exc: BaseException) -> ArtifactPublicationError:
    return ArtifactPublicationError(
        f"Artifact publication failed: kind={artifact.kind.value} {_owner(artifact)} "
        f"path={artifact.path} stage={stage}: {exc}"
    )


@contextmanager
def _temporary_text(artifact: PlannedArtifact, *, newline: str | None = None) -> Iterator[IO[str]]:
    path = artifact.path
    if not path.is_absolute():
        raise ArtifactPublicationError(
            f"Artifact path must be absolute: kind={artifact.kind.value} {_owner(artifact)} path={path}"
        )
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(raw)
        with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except ArtifactPublicationError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise _error(artifact, "write", exc) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_text(artifact: PlannedArtifact, *, content: str) -> None:
    with _temporary_text(artifact, newline="") as handle:
        handle.write(content)


def publish_json(artifact: PlannedArtifact, *, data: object) -> None:
    try:
        content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    except (TypeError, ValueError) as exc:
        raise _error(artifact, "serialize-json", exc) from exc
    publish_text(artifact, content=content)


@contextmanager
def open_csv_artifact(
    artifact: PlannedArtifact,
    *,
    fieldnames: Sequence[str],
    delimiter: str = ",",
) -> Iterator[csv.DictWriter]:
    with _temporary_text(artifact, newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter=delimiter,
            lineterminator="\n",
        )
        yield writer
