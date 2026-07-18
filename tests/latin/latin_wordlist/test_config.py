from pathlib import Path

import pytest

from nlpo_toolkit.latin.latin_wordlist.config import (
    LatinWordlistConfig,
    load_wordlist_build_request,
)
from nlpo_toolkit.latin.latin_wordlist.errors import LatinWordlistConfigError


def test_defaults_and_relative_paths_are_resolved_from_config(tmp_path: Path) -> None:
    path = tmp_path / "wordlist.yml"
    path.write_text("{}\n", encoding="utf-8")
    request = load_wordlist_build_request(path)
    assert request.conllu_dir == (tmp_path / "input/treebank_latin").resolve()
    assert request.extra_wordlists == ()
    assert request.filters.min_text_freq == 3


@pytest.mark.parametrize(
    "yaml",
    (
        "unknown: true\n",
        "inputs:\n  unknown: true\n",
        "filters:\n  min_length: true\n",
        "filters:\n  min_length: 0\n",
        "inputs:\n  conllu_dir: ''\n",
        "inputs:\n  extra_wordlists: [a.txt, a.txt]\n",
        "filters: {}\nfilters: {}\n",
        "- not-a-mapping\n",
    ),
)
def test_invalid_config_is_a_typed_error(tmp_path: Path, yaml: str) -> None:
    path = tmp_path / "wordlist.yml"
    path.write_text(yaml, encoding="utf-8")
    with pytest.raises(LatinWordlistConfigError):
        load_wordlist_build_request(path)


def test_empty_extra_punctuation_and_absolute_paths_are_supported(tmp_path: Path) -> None:
    absolute = (tmp_path / "treebank").resolve()
    path = tmp_path / "wordlist.yml"
    path.write_text(
        f"inputs:\n  conllu_dir: {absolute}\ntokenize:\n  extra_punct: ''\n",
        encoding="utf-8",
    )
    request = load_wordlist_build_request(path)
    assert request.conllu_dir == absolute
    assert request.tokenization.extra_punct == ""
    assert isinstance(LatinWordlistConfig().inputs.extra_wordlists, tuple)


def test_output_inside_text_input_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "wordlist.yml"
    path.write_text(
        "inputs:\n  latin_text_dir: texts\n"
        "output:\n  latin_wordlist_out: texts/words.txt\n",
        encoding="utf-8",
    )
    with pytest.raises(LatinWordlistConfigError, match="would be collected as input"):
        load_wordlist_build_request(path)


def test_duplicate_paths_are_checked_after_resolution(tmp_path: Path) -> None:
    path = tmp_path / "wordlist.yml"
    path.write_text(
        f"inputs:\n  extra_wordlists: [extra.txt, {tmp_path / 'extra.txt'}]\n",
        encoding="utf-8",
    )
    with pytest.raises(LatinWordlistConfigError, match="resolves to duplicate"):
        load_wordlist_build_request(path)
