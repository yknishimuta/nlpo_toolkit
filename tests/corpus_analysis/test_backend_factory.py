from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

import nlpo_toolkit.corpus_analysis.runner as runner_mod
from nlpo_toolkit.backends import (
    BuiltNLPBackend,
    NLPBackendConfigError,
    NLPBackendInfo,
    create_nlp_backend,
)
from nlpo_toolkit.backends.factory import render_backend_info
from nlpo_toolkit.backends.stanza_backend import StanzaBackend
from nlpo_toolkit.backends.transformers_backend import (
    NLPBackendUnavailableError,
    TransformersBackend,
)
from nlpo_toolkit.corpus_analysis.config import NLPConfig, load_config
from nlpo_toolkit.corpus_analysis.features import FeatureOptions, build_feature_rows
from nlpo_toolkit.models import NLPDocument, NLPSentence, NLPToken


class FakeBackend:
    def __call__(self, text: str) -> NLPDocument:
        return NLPDocument(
            sentences=[
                NLPSentence(
                    text=text,
                    tokens=[
                        NLPToken(text="xiv", lemma="xiv", upos="NOUN", start_char=0),
                        NLPToken(text="amat", lemma="amo", upos="VERB", start_char=4),
                    ],
                )
            ],
            text=text,
        )


def test_fake_backend_works_for_count_and_features() -> None:
    from nlpo_toolkit.nlp import count_nouns

    backend = FakeBackend()

    assert count_nouns("ignored", backend, use_lemma=True) == Counter({"xiv": 1})
    rows = build_feature_rows(
        [("group_a", [Path("a.txt")], "ignored")],
        backend,
        FeatureOptions(),
    )
    assert rows[0]["token_count"] == 2


def test_factory_selects_stanza_without_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeStanzaBackend:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def __call__(self, text: str) -> NLPDocument:
            return NLPDocument(text=text)

    import nlpo_toolkit.backends.stanza_backend as stanza_mod

    monkeypatch.setattr(stanza_mod, "StanzaBackend", FakeStanzaBackend)

    built = create_nlp_backend(
        NLPConfig(
            backend="stanza",
            language="la",
            stanza_package="perseus",
            cpu_only=True,
        )
    )

    assert isinstance(built.backend, FakeStanzaBackend)
    assert calls == [
        {
            "lang": "la",
            "package": "perseus",
            "use_gpu": False,
            "processors": "tokenize,mwt,pos,lemma",
        }
    ]
    assert built.info == NLPBackendInfo(
        name="stanza",
        language="la",
        package="perseus",
        use_gpu=False,
    )


def test_factory_selects_transformers_without_stanza(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeTransformersBackend:
        def __init__(self, model_name: str):
            calls.append(model_name)

        def __call__(self, text: str) -> NLPDocument:
            return NLPDocument(text=text)

    import nlpo_toolkit.backends.transformers_backend as transformers_mod

    monkeypatch.setattr(transformers_mod, "TransformersBackend", FakeTransformersBackend)

    built = create_nlp_backend(
        NLPConfig(
            backend="transformers",
            language="la",
            model_name="example/model",
            cpu_only=True,
        )
    )

    assert isinstance(built.backend, FakeTransformersBackend)
    assert calls == ["example/model"]
    assert built.info.to_dict() == {
        "backend": "transformers",
        "language": "la",
        "package": None,
        "model": "example/model",
        "device": "cpu",
    }


def test_factory_rejects_transformers_without_model_name() -> None:
    with pytest.raises(NLPBackendConfigError, match="nlp.model_name is required"):
        create_nlp_backend(NLPConfig(backend="transformers", language="la"))


def test_config_rejects_transformers_without_model_name(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yml"
    cfg.write_text(
        "\n".join(
            [
                "groups:",
                "  group_a: {files: [input/a.txt]}",
                "nlp:",
                "  backend: transformers",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="nlp.model_name is required"):
        load_config(cfg)


def test_stanza_backend_conversion_preserves_common_model_fields() -> None:
    backend = StanzaBackend.__new__(StanzaBackend)
    word_a = SimpleNamespace(
        text="Rosa",
        lemma="rosa",
        upos="NOUN",
        start_char=0,
        end_char=4,
    )
    word_b = SimpleNamespace(
        text="amat",
        lemma="amo",
        upos="VERB",
        start_char=5,
        end_char=9,
    )
    stanza_doc = SimpleNamespace(
        sentences=[
            SimpleNamespace(text="Rosa amat", words=[word_a, word_b]),
            SimpleNamespace(text="", words=[]),
        ]
    )

    doc = backend._convert_to_common_model(stanza_doc, "Rosa amat")

    assert isinstance(doc, NLPDocument)
    assert len(doc.sentences) == 2
    assert doc.sentences[0].text == "Rosa amat"
    assert doc.sentences[0].tokens[0] == NLPToken(
        text="Rosa",
        lemma="rosa",
        upos="NOUN",
        start_char=0,
        end_char=4,
    )
    assert doc.sentences[0].tokens[1].lemma == "amo"
    assert doc.sentences[1].tokens == []


def test_stanza_backend_conversion_flattens_mwt_without_duplication() -> None:
    backend = StanzaBackend.__new__(StanzaBackend)
    word = SimpleNamespace(text="multi", lemma="multi", upos="NOUN")
    token = SimpleNamespace(words=[word])
    stanza_doc = SimpleNamespace(
        sentences=[SimpleNamespace(text="multi", words=[], tokens=[token])]
    )

    doc = backend._convert_to_common_model(stanza_doc, "multi")

    assert [tok.text for tok in doc.sentences[0].tokens] == ["multi"]


def test_transformers_backend_returns_common_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_pipeline(task: str, model: str):
        assert task == "token-classification"
        assert model == "example/model"
        return lambda text: [
            {
                "word": "Rosa",
                "entity": "B-NOUN",
                "start": 0,
                "end": 4,
            }
        ]

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(pipeline=fake_pipeline),
    )

    backend = TransformersBackend("example/model")
    doc = backend("Rosa")

    assert isinstance(doc, NLPDocument)
    assert doc.sentences[0].tokens[0] == NLPToken(
        text="Rosa",
        lemma="rosa",
        upos="NOUN",
        start_char=0,
        end_char=4,
    )


def test_transformers_backend_reports_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(NLPBackendUnavailableError, match="optional 'transformers' dependency"):
        TransformersBackend("example/model")


def test_runner_uses_backend_factory_and_records_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("ignored", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy", encoding="utf-8")

    def load_config_fn(_path: Path):
        return {
            "out_dir": "output",
            "groups": {"group_a": {"files": ["input/a.txt"]}},
        }

    def fake_factory(config: NLPConfig) -> BuiltNLPBackend:
        return BuiltNLPBackend(
            backend=FakeBackend(),
            info=NLPBackendInfo(name="fake", language=config.language),
        )

    monkeypatch.setattr(runner_mod, "run_preprocess_if_needed", lambda **kwargs: None)

    rc = runner_mod.run(
        project_root=tmp_path,
        config_path=config_path,
        load_config_fn=load_config_fn,
        clean_mod=object(),
        backend_factory=fake_factory,
        count_group_fn=runner_mod.__dict__.get("count_group_fn", None) or _count_group,
    )

    assert rc == 0
    csv_text = (tmp_path / "output" / "noun_frequency_group_a.csv").read_text(
        encoding="utf-8"
    )
    assert "xiv,1" in csv_text
    meta_text = (tmp_path / "output" / "run_meta.json").read_text(encoding="utf-8")
    assert '"backend": "fake"' in meta_text


def _count_group(text: str, nlp, **kwargs):
    from nlpo_toolkit.corpus_analysis.nlp_hooks import count_group

    return count_group(text, nlp, **kwargs)


def test_render_backend_info_is_generic() -> None:
    assert render_backend_info(
        NLPBackendInfo(name="transformers", language="la", model="example/model")
    ) == [
        "backend=transformers",
        "language=la",
        "device=cpu",
        "model=example/model",
    ]
