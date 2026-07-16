from __future__ import annotations

from pathlib import Path

import pytest

from nlpo_toolkit.configuration.yaml_loader import (
    YamlErrorKind, YamlLoadError, load_yaml_mapping,
)


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("suffix", (".yml", ".yaml"))
def test_loads_nested_mapping_and_sequence(suffix: str, tmp_path: Path) -> None:
    path = write(tmp_path / f"config{suffix}", "root:\n  nested: value\nitems:\n  - name: one\n")
    assert load_yaml_mapping(path) == {
        "root": {"nested": "value"}, "items": [{"name": "one"}]
    }


def test_empty_file_is_empty_mapping(tmp_path: Path) -> None:
    assert load_yaml_mapping(write(tmp_path / "empty.yml", "")) == {}


def test_missing_file_is_read_error(tmp_path: Path) -> None:
    with pytest.raises(YamlLoadError) as caught:
        load_yaml_mapping(tmp_path / "missing.yml")
    assert caught.value.kind is YamlErrorKind.READ
    assert caught.value.source_path.is_absolute()


def test_unreadable_file_is_read_error(tmp_path: Path, monkeypatch) -> None:
    path = write(tmp_path / "config.yml", "key: value\n")
    original = Path.read_text

    def fail(self: Path, *args, **kwargs):
        if self == path:
            raise PermissionError("denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail)
    with pytest.raises(YamlLoadError) as caught:
        load_yaml_mapping(path)
    assert caught.value.kind is YamlErrorKind.READ


def test_invalid_utf8_is_utf8_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_bytes(b"key: \xff")
    with pytest.raises(YamlLoadError) as caught:
        load_yaml_mapping(path)
    assert caught.value.kind is YamlErrorKind.UTF8


def test_invalid_yaml_is_parse_error(tmp_path: Path) -> None:
    with pytest.raises(YamlLoadError) as caught:
        load_yaml_mapping(write(tmp_path / "bad.yml", "key: [\n"))
    assert caught.value.kind is YamlErrorKind.PARSE
    assert ":1:" in str(caught.value)


@pytest.mark.parametrize("text", ("- item\n", "scalar\n"))
def test_non_mapping_root_is_rejected(text: str, tmp_path: Path) -> None:
    with pytest.raises(YamlLoadError) as caught:
        load_yaml_mapping(write(tmp_path / "root.yml", text))
    assert caught.value.kind is YamlErrorKind.ROOT_TYPE


@pytest.mark.parametrize("text", ("1: value\n", "null: value\n"))
def test_non_string_mapping_key_is_rejected(text: str, tmp_path: Path) -> None:
    with pytest.raises(YamlLoadError) as caught:
        load_yaml_mapping(write(tmp_path / "key.yml", text))
    assert caught.value.kind is YamlErrorKind.KEY_TYPE


@pytest.mark.parametrize(("text", "key"), (
    ("key: one\nkey: two\n", "key"),
    ("outer:\n  key: one\n  key: two\n", "key"),
    ("items:\n  - key: one\n    key: two\n", "key"),
    ("key: one\nkey: two\nkey: three\n", "key"),
))
def test_duplicate_keys_are_hard_errors_at_every_depth(
    text: str, key: str, tmp_path: Path
) -> None:
    path = write(tmp_path / "duplicate.yml", text)
    with pytest.raises(YamlLoadError) as caught:
        load_yaml_mapping(path)
    error = caught.value
    assert error.kind is YamlErrorKind.DUPLICATE_KEY
    assert key in str(error)
    assert str(path.resolve()) in str(error)
    assert any(f":{line}:" in str(error) for line in range(2, 5))


def test_unsafe_python_tag_is_rejected(tmp_path: Path) -> None:
    path = write(
        tmp_path / "unsafe.yml",
        'value: !!python/object/apply:os.system ["echo unsafe"]\n',
    )
    with pytest.raises(YamlLoadError) as caught:
        load_yaml_mapping(path)
    assert caught.value.kind is YamlErrorKind.PARSE


def test_merge_key_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path / "merge.yml", "base: &base\n  a: 1\nitem:\n  <<: *base\n")
    with pytest.raises(YamlLoadError, match="merge key"):
        load_yaml_mapping(path)
