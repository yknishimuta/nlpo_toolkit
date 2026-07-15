from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml

from nlpo_toolkit.cleaner_contracts import (
    CleanerConfig,
    CleanerConfigInspection,
    CleanerConfigParseError,
    CleanerConfigReadError,
    CleanerConfigValidationError,
    CLEANER_KINDS,
    CleanerKind,
    CleanerReferencedFile,
)


def _resolve_path(config_path: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _required_string(raw: Mapping[str, object], key: str, path: Path) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CleanerConfigValidationError(
            f"'{key}' is required in clean config YAML: {path}"
        )
    return value


def _optional_string(raw: Mapping[str, object], key: str, path: Path) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CleanerConfigValidationError(f"'{key}' must be a string: {path}")
    return value


def load_cleaner_config(path: str | Path) -> CleanerConfig:
    source_path = Path(path).expanduser().resolve()
    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CleanerConfigReadError(
            f"Failed to read cleaner config: {source_path}: {exc}"
        ) from exc
    except UnicodeError as exc:
        raise CleanerConfigReadError(
            f"Cleaner config is not valid UTF-8: {source_path}: {exc}"
        ) from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise CleanerConfigParseError(
            f"Invalid cleaner YAML: {source_path}: {exc}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise CleanerConfigValidationError(
            f"Cleaner config root must be a mapping: {source_path}"
        )

    kind = _required_string(raw, "kind", source_path)
    if kind not in CLEANER_KINDS:
        raise CleanerConfigValidationError(
            "'kind' must be one of: corpus_corporum, scholastic_text: "
            f"{source_path}"
        )
    input_raw = _required_string(raw, "input", source_path)
    output_raw = _required_string(raw, "output", source_path)

    def optional_path(key: str) -> Path | None:
        value = _optional_string(raw, key, source_path)
        return _resolve_path(source_path, value) if value else None

    return CleanerConfig(
        source_path=source_path,
        kind=cast(CleanerKind, kind),
        input_path=_resolve_path(source_path, input_raw),
        output_path=_resolve_path(source_path, output_raw),
        rules_path=optional_path("rules_path"),
        lexicon_map_path=optional_path("lexicon_map_path"),
        ref_tsv_path=optional_path("ref_tsv"),
        output_filename_template=_optional_string(
            raw, "output_filename_template", source_path
        ),
        doc_id_prefix=_optional_string(raw, "doc_id_prefix", source_path),
    )


def resolve_cleaner_input_files(input_path: Path) -> tuple[Path, ...]:
    path = input_path.resolve()
    if path.is_file():
        return (path,)
    if path.is_dir():
        return tuple(
            candidate.resolve()
            for candidate in sorted(path.iterdir())
            if candidate.is_file() and candidate.suffix.lower() == ".txt"
        )
    raise CleanerConfigValidationError(f"Cleaner input does not exist: {path}")


def inspect_cleaner_config(path: str | Path) -> CleanerConfigInspection:
    config = load_cleaner_config(path)
    references = tuple(
        CleanerReferencedFile(kind, reference)
        for kind, reference in (
            ("preprocess.rules_path", config.rules_path),
            ("preprocess.lexicon_map_path", config.lexicon_map_path),
            ("preprocess.ref_tsv", config.ref_tsv_path),
        )
        if reference is not None
    )
    return CleanerConfigInspection(
        config=config,
        input_files=resolve_cleaner_input_files(config.input_path),
        referenced_files=references,
    )
