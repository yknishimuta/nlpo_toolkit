from __future__ import annotations

from tests.corpus_analysis.fake_nlp import corpus_request

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import nlpo_toolkit.corpus_analysis.runner as runner_mod
from nlpo_toolkit.backends import (
    TransformersBackend as PublicTransformersBackend,
    create_nlp_backend,
)
from nlpo_toolkit.nlp.contracts import (
    BuiltNLPBackend,
    NLPBackendInfo,
    NLPBackendSpec,
    NLPDocument,
    NLPSentence,
    NLPToken,
)
from nlpo_toolkit.backends.factory import render_backend_info
from nlpo_toolkit.backends.stanza_backend import (
    convert_stanza_document_to_common_model,
)
from nlpo_toolkit.backends.transformers_backend import (
    NLPBackendUnavailableError,
    TransformersBackend,
)
from nlpo_toolkit.corpus_analysis.config import NLPConfig, load_config
from nlpo_toolkit.corpus_analysis.features import FeatureOptions, build_feature_rows
from tests.corpus_analysis.fake_nlp import runner_dependencies


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


def test_transformers_backend_remains_public() -> None:
    assert PublicTransformersBackend is TransformersBackend


def test_transformers_latin_adapter_is_not_exported() -> None:
    import nlpo_toolkit.backends as backends

    legacy_name = "Transformers" + "LatinAdapter"

    assert legacy_name not in backends.__all__
    assert not hasattr(backends, legacy_name)


def test_transformers_backend_module_has_no_legacy_adapter() -> None:
    import nlpo_toolkit.backends.transformers_backend as module

    legacy_name = "Transformers" + "LatinAdapter"

    assert not hasattr(module, legacy_name)


def test_backend_package_imports_without_legacy_adapter() -> None:
    import nlpo_toolkit.backends as backends

    assert hasattr(backends, "TransformersBackend")
    assert hasattr(backends, "create_nlp_backend")


def test_fake_backend_works_for_features() -> None:
    backend = FakeBackend()

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
        NLPBackendSpec(
            backend="stanza",
            language="la",
            stanza_package="perseus",
            use_gpu=False,
        ),
        processors=("tokenize", "mwt", "pos", "lemma"),
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
        NLPBackendSpec(
            backend="transformers",
            language="la",
            model_name="example/model",
            use_gpu=False,
        ),
        processors=("tokenize", "mwt", "pos", "lemma"),
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
    with pytest.raises(ValidationError, match="model_name is required"):
        NLPConfig(backend="transformers", language="la")


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

    with pytest.raises(ValueError, match="nlp.*model_name is required"):
        load_config(cfg)


def test_stanza_backend_conversion_preserves_common_model_fields() -> None:
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

    doc = convert_stanza_document_to_common_model(stanza_doc, "Rosa amat")

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
    assert doc.sentences[1].tokens == ()


def test_stanza_backend_conversion_flattens_mwt_without_duplication() -> None:
    word = SimpleNamespace(text="multi", lemma="multi", upos="NOUN")
    token = SimpleNamespace(words=[word])
    stanza_doc = SimpleNamespace(
        sentences=[SimpleNamespace(text="multi", words=[], tokens=[token])]
    )

    doc = convert_stanza_document_to_common_model(stanza_doc, "multi")

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

    factory_calls: list[NLPConfig] = []

    def fake_factory(config: NLPConfig) -> BuiltNLPBackend:
        factory_calls.append(config)
        return BuiltNLPBackend(
            backend=FakeBackend(),
            info=NLPBackendInfo(name="fake", language=config.language),
        )


    rc = runner_mod.run(
        corpus_request(tmp_path, config_path),
        dependencies=runner_dependencies(load_config_fn, fake_factory),
    )

    assert rc.exit_code == 0
    assert len(factory_calls) == 1
    assert factory_calls[0].language == "la"
    csv_text = (tmp_path / "output" / "frequency_group_a.csv").read_text(
        encoding="utf-8"
    )
    assert "xiv,1" in csv_text
    meta_text = (tmp_path / "output" / "run_meta.json").read_text(encoding="utf-8")
    assert '"backend": "fake"' in meta_text


def test_backend_factory_failure_has_no_fallback(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("ignored", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy", encoding="utf-8")

    def load_config_fn(_path: Path):
        return {
            "groups": {"group_a": {"files": ["input/a.txt"]}},
        }

    def failing_factory(_config: NLPConfig) -> BuiltNLPBackend:
        raise RuntimeError("backend initialization failed")

    with pytest.raises(RuntimeError, match="backend initialization failed"):
        runner_mod.run(
            corpus_request(tmp_path, config_path),
            dependencies=runner_dependencies(load_config_fn, failing_factory),
        )

def test_render_backend_info_is_generic() -> None:
    assert render_backend_info(
        NLPBackendInfo(name="transformers", language="la", model="example/model")
    ) == [
        "backend=transformers",
        "language=la",
        "device=cpu",
        "model=example/model",
    ]
