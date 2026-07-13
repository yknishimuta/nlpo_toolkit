from __future__ import annotations

from tests.corpus_analysis.fake_nlp import corpus_request
from pathlib import Path

from nlpo_toolkit.corpus_analysis.dictcheck import split_frequency_csv
from nlpo_toolkit.corpus_analysis.runner import run
from tests.corpus_analysis.fake_nlp import fake_backend_factory, runner_dependencies


def test_dictcheck_applies_lemma_normalize_map(tmp_path: Path):
    # wordlist contains normalized lemma
    wordlist = tmp_path / "latin_words.txt"
    wordlist.write_text("materia\n", encoding="utf-8")

    # freq csv contains unnormalized lemma
    freq = tmp_path / "frequency.csv"
    freq.write_text("word,frequency\nmaterium,10\n", encoding="utf-8")

    # normalize tsv maps materium -> materia
    norm = tmp_path / "lemma_normalize.tsv"
    norm.write_text("materium\tmateria\n", encoding="utf-8")

    known = tmp_path / "known.csv"
    unknown = tmp_path / "unknown.csv"

    k, u = split_frequency_csv(
        freq_csv=freq,
        wordlist_path=wordlist,
        out_known_csv=known,
        out_unknown_csv=unknown,
        lemma_col="word",
        count_col="frequency",
        normalize=True,
        normalize_map_path=norm,
    )

    assert k == 1
    assert u == 0


def test_runner_loads_planned_lemma_normalization_path(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("materium", encoding="utf-8")
    normalization = tmp_path / "config" / "lemma.tsv"
    normalization.parent.mkdir()
    normalization.write_text("materium\tmateria\n", encoding="utf-8")
    config_path = tmp_path / "groups.yml"
    config_path.write_text("dummy", encoding="utf-8")

    import nlpo_toolkit.corpus_analysis.dictcheck as dictcheck_mod

    original_loader = dictcheck_mod.load_lemma_normalize_map
    loaded_paths: list[Path] = []

    def recording_loader(path: Path):
        loaded_paths.append(path)
        return original_loader(path)

    monkeypatch.setattr(dictcheck_mod, "load_lemma_normalize_map", recording_loader)
    result = run(
        corpus_request(tmp_path, config_path),
        dependencies=runner_dependencies(
            lambda _path: {
                "groups": {"text": {"files": ["input/a.txt"]}},
                "out_dir": "output",
                "dictcheck": {"lemma_normalize": "config/lemma.tsv"},
            },
            fake_backend_factory([("materium", "materium", "NOUN")]),
        ),
    )

    assert result.exit_code == 0
    assert loaded_paths == [normalization.resolve()]
