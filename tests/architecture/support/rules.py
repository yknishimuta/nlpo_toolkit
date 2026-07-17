from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class DependencyRule:
    name: str
    source_prefixes: tuple[str, ...]
    forbidden_target_prefixes: tuple[str, ...]
    allowed_target_prefixes: tuple[str, ...] = ()
    excluded_source_prefixes: tuple[str, ...] = ()
    explanation: str = ""


@dataclass(frozen=True)
class ArchitectureViolation:
    rule_name: str
    importer: str
    imported: str
    source_path: Path
    line_number: int
    explanation: str = ""

    def __str__(self) -> str:
        detail = (
            f"[{self.rule_name}]\n{self.importer}\n"
            f"imports {self.imported}\nat {self.source_path}:{self.line_number}"
        )
        return f"{detail}\n\n{self.explanation}" if self.explanation else detail


@dataclass(frozen=True)
class DynamicImportAllowance:
    module_prefix: str
    reason: str
    target_prefix: str | None = None


class HasLocation(Protocol):
    source_path: Path
    line_number: int


def matches_prefix(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def format_violations(violations: object) -> str:
    ordered = sorted(violations, key=lambda item: str(item))  # type: ignore[arg-type]
    if not ordered:
        return ""
    return f"{len(ordered)} architecture violation(s)\n\n" + "\n\n".join(
        str(item) for item in ordered
    )
