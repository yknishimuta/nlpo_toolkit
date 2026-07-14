from pathlib import Path

import pytest

from nlpo_toolkit.cleaner_contracts import (
    CleanerConfig,
    CleanerConfigInspection,
    CleanerReferencedFile,
)
from nlpo_toolkit.corpus_analysis.config import ensure_app_config
from nlpo_toolkit.corpus_analysis.config_references import (
    ConfigFileReference,
    ConfigArchivePolicy,
    ConfigReferenceError,
    ResolvedConfigFiles,
    resolve_config_files,
)


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _inspection(
    *, cleaner: Path, referenced_files: tuple[CleanerReferencedFile, ...]
) -> CleanerConfigInspection:
    return CleanerConfigInspection(
        config=CleanerConfig(
            source_path=cleaner,
            kind="scholastic_text",
            input_path=cleaner.parent,
            output_path=cleaner.parent / "cleaned",
        ),
        input_files=(),
        referenced_files=referenced_files,
    )


def test_resolves_all_typed_references_and_snapshot_policy(tmp_path: Path) -> None:
    root_config = _touch(tmp_path / "config" / "groups.yml")
    cleaner = _touch(tmp_path / "config" / "cleaner.yml")
    rules = _touch(tmp_path / "config" / "rules.yml")
    lemma = _touch(tmp_path / "config" / "lemma.tsv")
    wordlist = _touch(tmp_path / "data" / "words.txt")
    patterns = _touch(tmp_path / "config" / "refs.txt")
    roman = _touch(tmp_path / "config" / "roman.txt")
    config = ensure_app_config(
        {
            "preprocess": {"kind": "cleaner", "config": "config/cleaner.yml"},
            "groups": {"text": {"files": []}},
            "dictcheck": {
                "enabled": False,
                "lemma_normalize": "config/lemma.tsv",
                "wordlist": "data/words.txt",
            },
            "ref_tags": {"enabled": False, "patterns": "config/refs.txt"},
            "filters": {"roman_exceptions_file": "config/roman.txt"},
        }
    )

    resolved = resolve_config_files(
        config=config,
        config_path=root_config,
        project_root=tmp_path,
        cleaner_inspection=_inspection(
            cleaner=cleaner,
            referenced_files=(CleanerReferencedFile("preprocess.rules_path", rules),),
        ),
    )

    assert [item.kind for item in resolved.references] == [
        "root_config",
        "preprocess.config",
        "preprocess.rules_path",
        "dictcheck.lemma_normalize",
        "dictcheck.wordlist",
        "ref_tags.patterns",
        "filters.roman_exceptions_file",
    ]
    assert resolved.require("root_config").source_path == root_config.resolve()
    assert resolved.path("preprocess.rules_path") == rules.resolve()
    assert resolved.path("dictcheck.lemma_normalize") == lemma.resolve()
    assert resolved.path("dictcheck.wordlist") == wordlist.resolve()
    assert resolved.path("ref_tags.patterns") == patterns.resolve()
    assert resolved.path("filters.roman_exceptions_file") == roman.resolve()
    assert resolved.require("root_config").archive_policy is ConfigArchivePolicy.SNAPSHOT
    assert resolved.require("dictcheck.wordlist").archive_policy is ConfigArchivePolicy.METADATA_ONLY
    assert resolved.require("dictcheck.wordlist").snapshot_relative_path is None


def test_absolute_path_and_external_snapshot_path(tmp_path: Path) -> None:
    project = tmp_path / "project"
    root_config = _touch(project / "groups.yml")
    external = _touch(tmp_path / "external" / "lemma.tsv")
    config = ensure_app_config(
        {
            "groups": {"text": {"files": []}},
            "dictcheck": {"lemma_normalize": str(external)},
        }
    )

    resolved = resolve_config_files(
        config=config,
        config_path=root_config,
        project_root=project,
        cleaner_inspection=None,
    )

    reference = resolved.require("dictcheck.lemma_normalize")
    assert reference.source_path == external.resolve()
    assert reference.snapshot_relative_path == Path("external/lemma.tsv")


def test_unset_optional_references_are_omitted(tmp_path: Path) -> None:
    root_config = _touch(tmp_path / "groups.yml")
    resolved = resolve_config_files(
        config=ensure_app_config({"groups": {"text": {"files": []}}}),
        config_path=root_config,
        project_root=tmp_path,
        cleaner_inspection=None,
    )
    assert [item.kind for item in resolved.references] == ["root_config"]
    assert resolved.get("dictcheck.wordlist") is None


def test_duplicate_kind_is_rejected(tmp_path: Path) -> None:
    path = _touch(tmp_path / "same.yml")
    reference = ConfigFileReference(
        kind="duplicate",
        source_path=path.resolve(),
        archive_policy=ConfigArchivePolicy.SNAPSHOT,
        snapshot_relative_path=Path("same.yml"),
    )
    with pytest.raises(ConfigReferenceError, match="duplicate"):
        ResolvedConfigFiles((reference, reference))


@pytest.mark.parametrize(
    "policy,snapshot_relative_path",
    [
        (ConfigArchivePolicy.SNAPSHOT, None),
        (ConfigArchivePolicy.SNAPSHOT, Path("/absolute.yml")),
        (ConfigArchivePolicy.SNAPSHOT, Path("../escape.yml")),
        (ConfigArchivePolicy.SNAPSHOT, Path(".")),
        (ConfigArchivePolicy.METADATA_ONLY, Path("unexpected.yml")),
    ],
)
def test_config_reference_rejects_invalid_archive_policy_combinations(
    tmp_path: Path,
    policy: ConfigArchivePolicy,
    snapshot_relative_path: Path | None,
) -> None:
    source = _touch(tmp_path / "source.yml").resolve()
    with pytest.raises(ValueError):
        ConfigFileReference(
            kind="test",
            source_path=source,
            archive_policy=policy,
            snapshot_relative_path=snapshot_relative_path,
        )


@pytest.mark.parametrize("make_path", [lambda path: path, lambda path: path.mkdir() or path])
def test_missing_or_directory_reference_is_rejected(tmp_path: Path, make_path) -> None:
    root_config = _touch(tmp_path / "groups.yml")
    configured = make_path(tmp_path / "configured")
    config = ensure_app_config(
        {
            "groups": {"text": {"files": []}},
            "dictcheck": {"lemma_normalize": str(configured)},
        }
    )
    expected = "does not exist" if not configured.exists() else "is not a file"

    with pytest.raises(ConfigReferenceError) as caught:
        resolve_config_files(
            config=config,
            config_path=root_config,
            project_root=tmp_path,
            cleaner_inspection=None,
        )

    message = str(caught.value)
    assert expected in message
    assert "dictcheck.lemma_normalize" in message
    assert str(configured.resolve()) in message


def test_optional_cleaner_entry_is_explicit_and_must_exist(tmp_path: Path) -> None:
    root_config = _touch(tmp_path / "groups.yml")
    cleaner = _touch(tmp_path / "cleaner.yml")
    missing = tmp_path / "optional.tsv"
    inspection = _inspection(
        cleaner=cleaner,
        referenced_files=(CleanerReferencedFile("preprocess.optional", missing, False),),
    )
    with pytest.raises(ConfigReferenceError, match="preprocess.optional"):
        resolve_config_files(
            config=ensure_app_config({"groups": {"text": {"files": []}}}),
            config_path=root_config,
            project_root=tmp_path,
            cleaner_inspection=inspection,
        )
