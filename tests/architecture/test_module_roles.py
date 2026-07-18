from .policy import (
    APPLICATION_MODULES,
    COMPOSITION,
    INFRASTRUCTURE_MODULES,
    MODULE_ROLE_POLICIES,
    PURE_MODULES,
    STANDALONE_CLI_MODULES,
    CLI,
    PORTS,
)
from .support.module_roles import (
    ModuleRole,
    find_module_role_issues,
    find_stale_role_selectors,
    format_module_role_issues,
    validate_role_policies,
    roles_for_module,
)
from .support.rules import matches_prefix


def test_classified_modules_do_not_have_multiple_primary_roles(
    production_graph,
) -> None:
    issues = tuple(
        issue
        for issue in find_module_role_issues(
            production_graph.modules, MODULE_ROLE_POLICIES
        )
        if issue.matched_roles
    )
    assert not issues, format_module_role_issues(issues)


def test_module_role_policy_has_no_stale_selectors(production_graph) -> None:
    stale = find_stale_role_selectors(production_graph.modules, MODULE_ROLE_POLICIES)
    assert not stale, format_module_role_issues(stale)


def test_module_role_policy_is_well_formed() -> None:
    problems = validate_role_policies(MODULE_ROLE_POLICIES)
    assert not problems, format_module_role_issues(problems)


def test_primary_roles_agree_with_existing_policy_groups(production_graph) -> None:
    expectations = (
        (APPLICATION_MODULES, ModuleRole.APPLICATION),
        (PURE_MODULES, ModuleRole.DOMAIN),
        (INFRASTRUCTURE_MODULES, ModuleRole.INFRASTRUCTURE),
        ((CLI, *STANDALONE_CLI_MODULES, COMPOSITION), ModuleRole.BOUNDARY),
        (
            (PORTS, "nlpo_toolkit.cleaner_contracts", "nlpo_toolkit.nlp.contracts"),
            ModuleRole.SHARED,
        ),
    )
    failures: list[str] = []
    for prefixes, expected in expectations:
        for prefix in prefixes:
            matched = tuple(
                module
                for module in production_graph.modules
                if matches_prefix(module, prefix)
            )
            if not matched:
                failures.append(f"stale consistency prefix: {prefix}")
            for module in matched:
                actual = roles_for_module(module, MODULE_ROLE_POLICIES)
                if actual and actual != {expected}:
                    failures.append(
                        f"{module}: expected={expected.value}; actual="
                        f"{','.join(sorted(role.value for role in actual)) or 'none'}"
                    )
    assert not failures, "\n".join(sorted(failures))
