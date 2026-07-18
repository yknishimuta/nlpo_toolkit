from __future__ import annotations

import csv
from pathlib import Path

from .errors import StylometryError
from .evaluation_models import AuthorshipAssignment, AuthorshipMetadata
from .models import InputFormat


def read_authorship_metadata(
    path: Path,
    *,
    input_format: InputFormat,
    id_column: str,
    author_column: str,
    work_column: str,
) -> AuthorshipMetadata:
    if not path.exists():
        raise StylometryError(f"authorship metadata file not found: {path}")
    if not path.is_file():
        raise StylometryError(f"authorship metadata path is not a file: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(
                csv.reader(stream, delimiter="," if input_format == "csv" else "\t")
            )
    except UnicodeDecodeError as exc:
        raise StylometryError(
            f"authorship metadata is not valid UTF-8: {path}"
        ) from exc
    except OSError as exc:
        raise StylometryError(
            f"could not read authorship metadata {path}: {exc}"
        ) from exc
    if not rows or not rows[0]:
        raise StylometryError("authorship metadata has no header")
    header = tuple(value.strip() for value in rows[0])
    if any(not name for name in header):
        raise StylometryError("authorship metadata contains an empty column name")
    if len(header) != len(set(header)):
        raise StylometryError("duplicate metadata column")
    for column, message in (
        (id_column, "metadata ID column not found"),
        (author_column, "author column not found"),
        (work_column, "work column not found"),
    ):
        if column not in header:
            raise StylometryError(f"{message}: {column}")
    indices = tuple(
        header.index(name) for name in (id_column, author_column, work_column)
    )
    assignments = []
    seen: dict[str, int] = {}
    work_authors: dict[str, str] = {}
    for row_number, row in enumerate(rows[1:], start=2):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != len(header):
            raise StylometryError(
                f"authorship metadata row {row_number} has the wrong width"
            )
        identifier, author, work = (row[index].strip() for index in indices)
        for value, name in (
            (identifier, "metadata ID"),
            (author, "author"),
            (work, "work ID"),
        ):
            if not value:
                raise StylometryError(f"{name} is empty at row {row_number}")
        previous = seen.get(identifier)
        if previous is not None:
            raise StylometryError(
                f"duplicate metadata ID {identifier!r} at rows {previous} and {row_number}"
            )
        seen[identifier] = row_number
        previous_author = work_authors.setdefault(work, author)
        if previous_author != author:
            raise StylometryError(f"work {work!r} is assigned to multiple authors")
        assignments.append(AuthorshipAssignment(identifier, author, work))
    return AuthorshipMetadata(tuple(assignments))
