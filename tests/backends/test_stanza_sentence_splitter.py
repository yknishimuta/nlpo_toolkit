from __future__ import annotations

from nlpo_toolkit.corpus_analysis import composition
from nlpo_toolkit.corpus_analysis.config import ensure_app_config


def test_composition_builds_tokenize_only_stanza_sentence_splitter(monkeypatch) -> None:
    calls = []
    sentinel = object()

    def fake_backend(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(composition, "StanzaBackend", fake_backend)
    cpu_config = ensure_app_config(
        {"groups": {"text": {"files": ["input.txt"]}}}
    ).nlp
    gpu_config = cpu_config.model_copy(update={"cpu_only": False, "stanza_package": "x"})

    assert composition._create_sentence_splitter(cpu_config) is sentinel
    assert composition._create_sentence_splitter(gpu_config) is sentinel
    assert calls == [
        {
            "lang": cpu_config.language,
            "package": cpu_config.stanza_package or "perseus",
            "use_gpu": False,
            "processors": "tokenize",
        },
        {
            "lang": gpu_config.language,
            "package": "x",
            "use_gpu": True,
            "processors": "tokenize",
        },
    ]
