from pathlib import Path

from nlpo_toolkit.latin.latin_wordlist.composition import (
    default_latin_wordlist_dependencies,
)
from nlpo_toolkit.latin.latin_wordlist.config import load_wordlist_build_request
from nlpo_toolkit.latin.latin_wordlist.service import execute_latin_wordlist_build


def test_small_corpus_end_to_end(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    texts = tmp_path / "texts"
    tree.mkdir()
    texts.mkdir()
    (tree / "sample.conllu").write_text(
        "1\tpuella\tpuella\n2\trosam\trosa\n3\tpuella\tpuella\n",
        encoding="utf-8",
    )
    (texts / "sample.txt").write_text("Deus deus deus", encoding="utf-8")
    (tmp_path / "extra.txt").write_text("homo\nbonus\n", encoding="utf-8")
    config = tmp_path / "config.yml"
    config.write_text(
        "inputs:\n  conllu_dir: tree\n  latin_text_dir: texts\n"
        "  extra_wordlists: [extra.txt]\n"
        "output:\n  latin_wordlist_out: output/words.txt\n",
        encoding="utf-8",
    )
    request = load_wordlist_build_request(config)
    result = execute_latin_wordlist_build(
        request, dependencies=default_latin_wordlist_dependencies()
    )
    assert result.word_count == 5
    assert request.output_path.read_text(encoding="utf-8") == (
        "bonus\ndeus\nhomo\npuella\nrosa\n"
    )
