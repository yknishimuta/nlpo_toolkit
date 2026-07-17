from pathlib import Path

from .policy import APPLICATION_MODULES, CLI, PURE_MODULES, STANDALONE_CLI_MODULES
from .support.rules import format_violations
from .support.source_checks import find_attribute_accesses, find_calls, find_imports


def _paths_for_modules(production_graph, modules, root: Path):
    selected = set()
    for edge in production_graph.edges:
        if any(edge.importer == item or edge.importer.startswith(item + ".") for item in modules):
            selected.add(edge.source_path)
    for module in production_graph.modules:
        if any(module == item or module.startswith(item + ".") for item in modules):
            rel = module.removeprefix("nlpo_toolkit").lstrip(".").replace(".", "/")
            path = root / f"{rel}.py"
            package_init = root / rel / "__init__.py"
            selected.add(package_init if package_init.exists() else path)
    return tuple(selected)


def test_pure_domain_has_no_io_or_terminal_access(production_graph) -> None:
    root = Path("nlpo_toolkit")
    paths = _paths_for_modules(production_graph, PURE_MODULES, root)
    violations = (
        find_calls(paths, qualified_names={"open", "pathlib.Path.open", "pathlib.Path.read_text", "pathlib.Path.write_text", "pathlib.Path.mkdir", "pathlib.Path.replace", "os.remove", "print"}, rule_name="pure-no-io")
        + find_imports(paths, module_prefixes={"argparse", "shutil", "subprocess"}, rule_name="pure-no-io-library")
        + find_attribute_accesses(paths, qualified_names={"sys.stdout", "sys.stderr"}, rule_name="pure-no-terminal")
    )
    assert not violations, format_violations(violations)


def test_application_services_have_no_terminal_api(production_graph) -> None:
    paths = _paths_for_modules(production_graph, APPLICATION_MODULES, Path("nlpo_toolkit"))
    violations = (
        find_calls(paths, qualified_names={"print"}, rule_name="application-no-terminal")
        + find_imports(paths, module_prefixes={"argparse"}, rule_name="application-no-argparse")
        + find_attribute_accesses(paths, qualified_names={"sys.stdout", "sys.stderr"}, rule_name="application-no-terminal")
    )
    assert not violations, format_violations(violations)


def test_argparse_is_limited_to_cli_adapters(production_graph, production_paths) -> None:
    cli_paths = _paths_for_modules(
        production_graph, (CLI, *STANDALONE_CLI_MODULES), Path("nlpo_toolkit")
    )
    paths = tuple(path for path in production_paths if path not in set(cli_paths))
    violations = find_imports(
        paths, module_prefixes={"argparse"}, rule_name="argparse-only-in-cli"
    )
    assert not violations, format_violations(violations)
