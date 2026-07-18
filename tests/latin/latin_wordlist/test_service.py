from dataclasses import dataclass, field
from pathlib import Path

import pytest

from nlpo_toolkit.latin.latin_wordlist.models import (
    ConlluCandidates,
    ExtraWordlistCandidates,
    LatinWordlistBuildRequest,
    TextCandidates,
    WordlistFilterPolicy,
    WordlistPublication,
    WordlistTokenizationPolicy,
)
from nlpo_toolkit.latin.latin_wordlist.ports import LatinWordlistDependencies
from nlpo_toolkit.latin.latin_wordlist.service import execute_latin_wordlist_build


@dataclass
class RecordingPublisher:
    calls: list[WordlistPublication] = field(default_factory=list)

    def __call__(self, publication: WordlistPublication) -> None:
        self.calls.append(publication)


def _request(tmp_path: Path) -> LatinWordlistBuildRequest:
    return LatinWordlistBuildRequest(
        tmp_path / "config.yml",
        tmp_path / "tree",
        tmp_path / "texts",
        (tmp_path / "z.txt", tmp_path / "a.txt"),
        tmp_path / "out.txt",
        WordlistFilterPolicy(2, 2, 3),
        WordlistTokenizationPolicy(""),
    )


def test_service_uses_ports_in_order_and_returns_typed_statistics(
    tmp_path: Path, capsys
) -> None:
    events: list[str] = []
    publisher = RecordingPublisher()

    def conllu(**kwargs):
        events.append("conllu")
        return ConlluCandidates((), {"lemma"}, {"forma": 2}), ()

    def text(**kwargs):
        events.append("text")
        return TextCandidates((), {"textus": 3}), ()

    def extra(*, path):
        events.append(path.name)
        return ExtraWordlistCandidates(path, {path.stem}), ()

    result = execute_latin_wordlist_build(
        _request(tmp_path),
        dependencies=LatinWordlistDependencies(conllu, text, extra, publisher),
    )
    assert events == ["conllu", "text", "z.txt", "a.txt"]
    assert publisher.calls[0].entries == ("a", "forma", "lemma", "textus", "z")
    assert result.word_count == 5
    assert result.statistics.extra_wordlist_counts == {
        (tmp_path / "z.txt").resolve(): 1,
        (tmp_path / "a.txt").resolve(): 1,
    }
    assert capsys.readouterr() == ("", "")


def test_service_propagates_publication_failure(tmp_path: Path) -> None:
    def fail(publication):
        raise RuntimeError("publisher failed")

    dependencies = LatinWordlistDependencies(
        lambda **kwargs: (ConlluCandidates((), (), {}), ()),
        lambda **kwargs: (TextCandidates((), {}), ()),
        lambda **kwargs: (None, ()),
        fail,
    )
    with pytest.raises(RuntimeError, match="publisher failed"):
        execute_latin_wordlist_build(_request(tmp_path), dependencies=dependencies)
