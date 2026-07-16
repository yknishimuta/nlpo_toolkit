from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml


class YamlErrorKind(str, Enum):
    READ = "read"
    UTF8 = "utf8"
    PARSE = "parse"
    DUPLICATE_KEY = "duplicate_key"
    ROOT_TYPE = "root_type"
    KEY_TYPE = "key_type"


@dataclass(frozen=True)
class YamlErrorDetails:
    source_path: Path
    kind: YamlErrorKind
    message: str


class YamlLoadError(ValueError):
    def __init__(self, details: YamlErrorDetails) -> None:
        self.details = details
        super().__init__(details.message)

    @property
    def source_path(self) -> Path:
        return self.details.source_path

    @property
    def kind(self) -> YamlErrorKind:
        return self.details.kind


class _StrictSafeLoader(yaml.SafeLoader):
    def __init__(self, stream: str, *, source_path: Path) -> None:
        super().__init__(stream)
        self.source_path = source_path


def _error(loader: _StrictSafeLoader, kind: YamlErrorKind, message: str) -> YamlLoadError:
    return YamlLoadError(YamlErrorDetails(loader.source_path, kind, message))


def _construct_mapping(
    loader: _StrictSafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key_node, value_node in node.value:
        line = key_node.start_mark.line + 1
        column = key_node.start_mark.column + 1
        if key_node.value == "<<":
            raise _error(
                loader,
                YamlErrorKind.PARSE,
                f"YAML merge key '<<' is not supported in "
                f"{loader.source_path}:{line}:{column}",
            )
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise _error(
                loader,
                YamlErrorKind.KEY_TYPE,
                f"YAML mapping key must be a string in "
                f"{loader.source_path}:{line}:{column}; got {type(key).__name__}",
            )
        if key in result:
            raise _error(
                loader,
                YamlErrorKind.DUPLICATE_KEY,
                f"Duplicate YAML key {key!r} in "
                f"{loader.source_path}:{line}:{column}",
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _parse_yaml_mapping(text: str, *, source_path: Path) -> dict[str, object]:
    loader = _StrictSafeLoader(text, source_path=source_path)
    try:
        try:
            raw = loader.get_single_data()
        except YamlLoadError:
            raise
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            location = (
                f":{mark.line + 1}:{mark.column + 1}" if mark is not None else ""
            )
            raise _error(
                loader,
                YamlErrorKind.PARSE,
                f"Invalid YAML in {source_path}{location}: {exc}",
            ) from exc
    finally:
        loader.dispose()
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise YamlLoadError(YamlErrorDetails(
            source_path,
            YamlErrorKind.ROOT_TYPE,
            f"Top-level YAML must be a mapping in {source_path}; "
            f"got {type(raw).__name__}",
        ))
    return raw


def load_yaml_mapping(path: str | Path) -> dict[str, object]:
    source_path = Path(path).expanduser().resolve()
    try:
        text = source_path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise YamlLoadError(YamlErrorDetails(
            source_path,
            YamlErrorKind.UTF8,
            f"YAML file is not valid UTF-8: {source_path}: {exc}",
        )) from exc
    except OSError as exc:
        raise YamlLoadError(YamlErrorDetails(
            source_path,
            YamlErrorKind.READ,
            f"Failed to read YAML file: {source_path}: {exc}",
        )) from exc
    return _parse_yaml_mapping(text, source_path=source_path)
