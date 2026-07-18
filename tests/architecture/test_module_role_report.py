from __future__ import annotations

import io

from .module_role_report import render_unclassified_modules_report, run_report
from .support.module_roles import (
    ModuleRole,
    ModuleRolePolicy,
    find_unclassified_modules,
)


POLICIES = (
    ModuleRolePolicy(
        role=ModuleRole.DOMAIN,
        exact_modules=("sample.domain",),
    ),
    ModuleRolePolicy(
        role=ModuleRole.APPLICATION,
        recursive_packages=("sample.application",),
    ),
)


def test_find_unclassified_modules_returns_stable_difference() -> None:
    modules = (
        "sample.z_unclassified",
        "sample.application.service",
        "sample.a_unclassified",
        "sample.domain",
        "sample.z_unclassified",
    )

    assert find_unclassified_modules(modules, POLICIES) == (
        "sample.a_unclassified",
        "sample.z_unclassified",
    )


def test_find_unclassified_modules_returns_empty_when_all_are_classified() -> None:
    assert (
        find_unclassified_modules(
            ("sample.domain", "sample.application.service"), POLICIES
        )
        == ()
    )


def test_report_renders_present_and_absent_unclassified_modules() -> None:
    assert render_unclassified_modules_report(("sample.z", "sample.a")) == (
        "Unclassified production modules:\n- sample.a\n- sample.z"
    )
    assert render_unclassified_modules_report(()) == (
        "No unclassified production modules."
    )


def test_report_exit_code_is_zero_with_or_without_unclassified_modules() -> None:
    for modules, expected in (
        (("sample.domain",), "No unclassified production modules."),
        (("sample.unclassified",), "- sample.unclassified"),
    ):
        stream = io.StringIO()

        assert run_report(modules, POLICIES, stream=stream) == 0
        assert expected in stream.getvalue()
