from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

from ..features.models import FeatureCommandResult, FeatureScalar
from .compare_rendering import render_csv_comparison_rows


@contextmanager
def open_cli_output(*, path: Path | None, stdout: TextIO) -> Iterator[TextIO]:
    if path is None:
        yield stdout
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        yield stream


def present_error(error: Exception, *, stderr: TextIO) -> None:
    print(f"[ERROR] {error}", file=stderr)


def write_mapping_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str],
    stream: TextIO,
    output_format: str,
) -> None:
    delimiter = "," if output_format == "csv" else "\t"
    writer = csv.DictWriter(stream, fieldnames=list(columns), delimiter=delimiter)
    writer.writeheader()
    writer.writerows(rows)


def write_feature_result(
    result: FeatureCommandResult,
    *,
    stream: TextIO,
    output_format: str,
) -> None:
    rows = list(result.rows)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    def format_value(value: FeatureScalar | str) -> FeatureScalar:
        if isinstance(value, float):
            if value.is_integer():
                return str(int(value))
            return f"{value:.12g}"
        return value

    formatted = tuple(
        {key: format_value(row.get(key, "")) for key in columns} for row in rows
    )
    write_mapping_rows(
        formatted,
        columns=columns,
        stream=stream,
        output_format=output_format,
    )


def write_ngram_result(result, *, stream: TextIO, output_format: str) -> None:
    columns = ["ngram", "count", "n", "field"]
    if result.by_group:
        columns.append("group")
    write_mapping_rows(
        tuple(row.as_mapping(by_group=result.by_group) for row in result.rows),
        columns=columns,
        stream=stream,
        output_format=output_format,
    )


def write_concordance_result(result, *, stream: TextIO, output_format: str) -> None:
    write_mapping_rows(
        result.rows,
        columns=result.columns,
        stream=stream,
        output_format=output_format,
    )


def write_compare_result(result, *, stream: TextIO, output_format: str) -> None:
    rendered = render_csv_comparison_rows(result)
    def format_value(value: Any) -> Any:
        if isinstance(value, float):
            if value.is_integer():
                return str(int(value))
            return f"{value:.12g}"
        return value

    rows = tuple(
        {key: format_value(value) for key, value in row.items()}
        for row in rendered.rows
    )
    write_mapping_rows(
        rows,
        columns=rendered.columns,
        stream=stream,
        output_format=output_format,
    )
