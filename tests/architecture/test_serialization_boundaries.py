from pathlib import Path

from .support.rules import format_violations
from .support.source_checks import find_calls


def test_only_common_yaml_loader_parses_yaml(production_paths) -> None:
    allowed = Path("nlpo_toolkit/configuration/yaml_loader.py")
    paths = tuple(path for path in production_paths if path != allowed)
    violations = find_calls(
        paths,
        qualified_names={"yaml.load", "yaml.safe_load", "yaml.full_load", "yaml.unsafe_load"},
        rule_name="yaml-parsing-owned-by-common-loader",
    )
    assert not violations, format_violations(violations)

