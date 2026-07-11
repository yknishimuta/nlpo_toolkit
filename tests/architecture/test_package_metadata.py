from pathlib import Path
import tomllib


def _project() -> dict[str, object]:
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_pyproject_contains_required_metadata() -> None:
    project = _project()
    assert project["name"] == "nlpo_toolkit"
    assert project["version"]
    assert project["description"]
    assert project["readme"] == "README.md"
    assert project["requires-python"] == ">=3.11"
    assert project["license"] == "MIT"
    assert project["scripts"]["nlpo"] == "nlpo_toolkit.corpus_analysis.cli:main"
    assert set(project["optional-dependencies"]["dev"]) == {"build", "pytest", "ruff"}


def test_requirements_file_is_not_a_second_dependency_source() -> None:
    assert not Path("requirements.txt").exists()


def test_version_is_not_hardcoded_in_python_modules() -> None:
    version = str(_project()["version"])
    offenders = [path for path in Path("nlpo_toolkit").rglob("*.py") if version in path.read_text(encoding="utf-8")]
    assert offenders == []


def test_core_runtime_dependencies_are_declared() -> None:
    dependencies = [str(item).lower() for item in _project()["dependencies"]]
    assert any(item.startswith("stanza>=1.9,<1.10") for item in dependencies)
    assert any(item.startswith("pyyaml>=6.0") for item in dependencies)
