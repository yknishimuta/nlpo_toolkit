from pathlib import Path

from .support.rules import format_violations
from .support.source_checks import find_calls, find_imports, find_method_calls


PACKAGE = Path("nlpo_toolkit/latin/latin_wordlist")


def test_latin_wordlist_io_and_presentation_are_limited_to_boundaries() -> None:
    paths = tuple(PACKAGE.glob("*.py"))
    cli = PACKAGE / "cli.py"
    adapters = {PACKAGE / "collectors.py", PACKAGE / "publication.py"}
    violations = (
        find_calls(
            (path for path in paths if path != cli),
            qualified_names={"print"},
            rule_name="latin-wordlist-print-only-cli",
        )
        + find_imports(
            (path for path in paths if path != cli),
            module_prefixes={"argparse"},
            rule_name="latin-wordlist-argparse-only-cli",
        )
        + find_method_calls(
            (path for path in paths if path not in adapters),
            method_names={"read_text", "write_text", "open", "rglob", "mkdir"},
            rule_name="latin-wordlist-io-only-adapters",
        )
        + find_imports(
            (path for path in paths if path.name != "config.py"),
            module_prefixes={"yaml", "nlpo_toolkit.configuration.yaml_loader"},
            rule_name="latin-wordlist-yaml-only-config",
        )
    )
    assert not violations, format_violations(violations)


def test_latin_wordlist_has_no_unsafe_exception_or_decoding_patterns() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "except Exception" in source:
            offenders.append(f"{path}: broad exception capture")
        if 'errors="ignore"' in source or "errors='ignore'" in source:
            offenders.append(f"{path}: ignored decoding error")
    assert not offenders, "\n".join(offenders)


def test_deleted_latin_wordlist_monolith_is_absent() -> None:
    assert not (PACKAGE / "build_latin_wordlist.py").exists()
