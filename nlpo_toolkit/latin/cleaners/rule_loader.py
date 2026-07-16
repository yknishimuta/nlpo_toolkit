from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from nlpo_toolkit.configuration.yaml_loader import YamlLoadError, load_yaml_mapping

from .errors import CleanerRuleConfigError
from .models import LineRemoveRule, RuleReference, RuleSet, SubstituteRule


def _string_field(raw: Mapping[object, object], key: str, *, path: Path, label: str, default: str = "") -> str:
    value = raw.get(key, default)
    if not isinstance(value, str):
        raise CleanerRuleConfigError(f"{path}: {label}.{key} must be a string")
    return value


def _parse_rule_reference(raw: object, *, path: Path, section: str, index: int) -> RuleReference:
    label = f"{section}[{index}]"
    if raw is None or raw == "":
        return RuleReference()
    if isinstance(raw, str):
        key = raw.strip()
        author, separator, work = key.partition(":")
        return RuleReference(key=key, author=author.strip() if separator else "", work=work.strip() if separator else "")
    if not isinstance(raw, Mapping):
        raise CleanerRuleConfigError(f"{path}: {label}.ref must be a string or mapping")
    allowed = {"key", "author", "work", "loc"}
    unknown = set(raw) - allowed
    if unknown:
        raise CleanerRuleConfigError(f"{path}: {label}.ref has unknown keys: {sorted(map(str, unknown))}")
    values = {key: _string_field(raw, key, path=path, label=f"{label}.ref") for key in allowed}
    key = values["key"].strip()
    author = values["author"].strip()
    work = values["work"].strip()
    if not key and (author or work):
        key = f"{author}:{work}".strip(":")
    return RuleReference(key, author, work, values["loc"].strip())


def _items(raw: Mapping[object, object], section: str, path: Path) -> list[object]:
    value = raw.get(section, [])
    if not isinstance(value, list):
        raise CleanerRuleConfigError(f"{path}: {section} must be a list")
    return value


def _compile(pattern: str, *, path: Path, section: str, index: int) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise CleanerRuleConfigError(f"{path}: invalid regex in {section}[{index}]: {pattern!r}: {exc}") from exc


def load_rule_set(path: str | Path) -> RuleSet:
    source = Path(path).resolve()
    try:
        raw = load_yaml_mapping(source)
    except YamlLoadError as exc:
        raise CleanerRuleConfigError(str(exc)) from exc
    unknown = set(raw) - {"remove_line_patterns", "substitute_patterns"}
    if unknown:
        raise CleanerRuleConfigError(f"{source}: unknown top-level keys: {sorted(map(str, unknown))}")
    removes: list[LineRemoveRule] = []
    for index, item in enumerate(_items(raw, "remove_line_patterns", source)):
        label = f"remove_line_patterns[{index}]"
        if not isinstance(item, Mapping):
            raise CleanerRuleConfigError(f"{source}: {label} must be a mapping")
        enabled = item.get("enabled", True)
        if not isinstance(enabled, bool):
            raise CleanerRuleConfigError(f"{source}: {label}.enabled must be a bool")
        if not enabled:
            continue
        pattern = _string_field(item, "pattern", path=source, label=label)
        if not pattern:
            raise CleanerRuleConfigError(f"{source}: {label}.pattern must be non-empty")
        removes.append(LineRemoveRule(_compile(pattern, path=source, section="remove_line_patterns", index=index), _parse_rule_reference(item.get("ref"), path=source, section="remove_line_patterns", index=index), _string_field(item, "name", path=source, label=label)))
    substitutions: list[SubstituteRule] = []
    for index, item in enumerate(_items(raw, "substitute_patterns", source)):
        label = f"substitute_patterns[{index}]"
        if not isinstance(item, Mapping):
            raise CleanerRuleConfigError(f"{source}: {label} must be a mapping")
        enabled = item.get("enabled", True)
        if not isinstance(enabled, bool):
            raise CleanerRuleConfigError(f"{source}: {label}.enabled must be a bool")
        if not enabled:
            continue
        pattern = _string_field(item, "pattern", path=source, label=label)
        if not pattern:
            raise CleanerRuleConfigError(f"{source}: {label}.pattern must be non-empty")
        substitutions.append(SubstituteRule(_compile(pattern, path=source, section="substitute_patterns", index=index), _string_field(item, "repl", path=source, label=label), _parse_rule_reference(item.get("ref"), path=source, section="substitute_patterns", index=index), _string_field(item, "name", path=source, label=label)))
    return RuleSet(tuple(removes), tuple(substitutions))
