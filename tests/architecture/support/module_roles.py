from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .rules import matches_prefix


class ModuleRole(Enum):
    SHARED = "shared"
    DOMAIN = "domain"
    APPLICATION = "application"
    INFRASTRUCTURE = "infrastructure"
    BOUNDARY = "boundary"


@dataclass(frozen=True)
class ModuleRolePolicy:
    role: ModuleRole
    exact_modules: tuple[str, ...] = ()
    recursive_packages: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModuleRoleIssue:
    module: str
    matched_roles: tuple[ModuleRole, ...]

    @property
    def kind(self) -> str:
        return (
            "unclassified-module"
            if not self.matched_roles
            else "multiply-classified-module"
        )

    def __str__(self) -> str:
        if not self.matched_roles:
            return (
                f"[{self.kind}]\n{self.module}\n\n"
                "Every production module must have exactly one primary architecture role.\n"
                "Add the narrowest appropriate exact module or recursive package selector\n"
                "to tests/architecture/policy.py."
            )
        roles = "\n".join(f"- {role.value}" for role in self.matched_roles)
        return (
            f"[{self.kind}]\n{self.module}\n\nmatched roles:\n{roles}\n\n"
            "A production module must have exactly one primary architecture role.\n"
            "Narrow or remove the overlapping selectors."
        )


@dataclass(frozen=True)
class StaleRoleSelector:
    role: ModuleRole
    selector_kind: str
    selector: str

    def __str__(self) -> str:
        return (
            "[stale-role-selector]\n"
            f"role: {self.role.value}\nkind: {self.selector_kind}\n"
            f"selector: {self.selector}"
        )


@dataclass(frozen=True)
class RolePolicyProblem:
    message: str

    def __str__(self) -> str:
        return f"[invalid-role-policy]\n{self.message}"


def roles_for_module(
    module: str, policies: tuple[ModuleRolePolicy, ...]
) -> frozenset[ModuleRole]:
    roles: set[ModuleRole] = set()
    for policy in policies:
        if module in policy.exact_modules or any(
            matches_prefix(module, package) for package in policy.recursive_packages
        ):
            roles.add(policy.role)
    return frozenset(roles)


def find_module_role_issues(
    modules: Iterable[str], policies: tuple[ModuleRolePolicy, ...]
) -> tuple[ModuleRoleIssue, ...]:
    issues = []
    for module in sorted(set(modules)):
        roles = roles_for_module(module, policies)
        if len(roles) != 1:
            issues.append(
                ModuleRoleIssue(
                    module, tuple(sorted(roles, key=lambda role: role.value))
                )
            )
    return tuple(issues)


def find_unclassified_modules(
    modules: Iterable[str], policies: tuple[ModuleRolePolicy, ...]
) -> tuple[str, ...]:
    return tuple(
        module
        for module in sorted(set(modules))
        if not roles_for_module(module, policies)
    )


def find_stale_role_selectors(
    modules: Iterable[str], policies: tuple[ModuleRolePolicy, ...]
) -> tuple[StaleRoleSelector, ...]:
    available = frozenset(modules)
    stale: list[StaleRoleSelector] = []
    for policy in policies:
        for module in policy.exact_modules:
            if module not in available:
                stale.append(StaleRoleSelector(policy.role, "exact", module))
        for package in policy.recursive_packages:
            if not any(matches_prefix(module, package) for module in available):
                stale.append(StaleRoleSelector(policy.role, "recursive", package))
    return tuple(
        sorted(
            stale, key=lambda item: (item.role.value, item.selector_kind, item.selector)
        )
    )


def validate_role_policies(
    policies: tuple[ModuleRolePolicy, ...],
) -> tuple[RolePolicyProblem, ...]:
    problems: list[RolePolicyProblem] = []
    selector_roles: dict[tuple[str, str], set[ModuleRole]] = {}
    for policy in policies:
        for kind, selectors in (
            ("exact", policy.exact_modules),
            ("recursive", policy.recursive_packages),
        ):
            if len(selectors) != len(set(selectors)):
                problems.append(
                    RolePolicyProblem(
                        f"duplicate {kind} selector within {policy.role.value} policy"
                    )
                )
            for selector in selectors:
                if not selector:
                    problems.append(
                        RolePolicyProblem(
                            f"empty {kind} selector in {policy.role.value} policy"
                        )
                    )
                elif not (
                    selector == "nlpo_toolkit" or selector.startswith("nlpo_toolkit.")
                ):
                    problems.append(
                        RolePolicyProblem(
                            f"selector must start with nlpo_toolkit: {selector}"
                        )
                    )
                if kind == "recursive" and (selector.endswith(".*") or "/" in selector):
                    problems.append(
                        RolePolicyProblem(
                            f"invalid recursive selector syntax: {selector}"
                        )
                    )
                if kind == "recursive" and selector in {
                    "nlpo_toolkit",
                    "nlpo_toolkit.corpus_analysis",
                    "nlpo_toolkit.comparison",
                    "nlpo_toolkit.corpus_analysis.artifacts",
                    "nlpo_toolkit.latin.cleaners",
                }:
                    problems.append(
                        RolePolicyProblem(
                            f"catch-all or heterogeneous recursive selector is forbidden: {selector}"
                        )
                    )
                selector_roles.setdefault((kind, selector), set()).add(policy.role)
    for (kind, selector), roles in selector_roles.items():
        if len(roles) > 1:
            names = ", ".join(sorted(role.value for role in roles))
            problems.append(
                RolePolicyProblem(
                    f"{kind} selector has multiple roles ({names}): {selector}"
                )
            )
    return tuple(sorted(set(problems), key=lambda item: item.message))


def format_module_role_issues(issues: Iterable[object]) -> str:
    def key(item: object) -> tuple[str, str]:
        if isinstance(item, ModuleRoleIssue):
            return (item.module, item.kind)
        if isinstance(item, StaleRoleSelector):
            return (item.selector, item.selector_kind)
        return (str(item), "")

    ordered = tuple(sorted(issues, key=key))
    return f"{len(ordered)} module role issue(s)\n\n" + "\n\n".join(
        str(issue) for issue in ordered
    )
