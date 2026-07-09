from __future__ import annotations

from nlpo_toolkit.corpus_analysis.outputs import build_frequency_output_paths


def test_build_frequency_output_paths() -> None:
    paths = build_frequency_output_paths("/tmp/output", "satyricon_cena")

    assert paths.base.name == "frequency_satyricon_cena.csv"
    assert paths.known.name == "frequency_satyricon_cena.known.csv"
    assert paths.unknown.name == "frequency_satyricon_cena.unknown.csv"
