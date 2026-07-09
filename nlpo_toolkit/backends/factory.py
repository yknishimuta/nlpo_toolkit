from __future__ import annotations

from dataclasses import dataclass

from nlpo_toolkit.count_vocabula.config import NLPConfig
from nlpo_toolkit.interfaces import NLPBackend


class NLPBackendConfigError(ValueError):
    pass


@dataclass(frozen=True)
class NLPBackendInfo:
    name: str
    language: str
    model: str | None = None
    package: str | dict[str, str] | None = None
    use_gpu: bool = False

    @property
    def device(self) -> str:
        return "gpu" if self.use_gpu else "cpu"

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "language": self.language,
            "package": self.package,
            "model": self.model,
            "device": self.device,
        }


@dataclass(frozen=True)
class BuiltNLPBackend:
    backend: NLPBackend
    info: NLPBackendInfo


def create_nlp_backend(config: NLPConfig) -> BuiltNLPBackend:
    if config.backend == "stanza":
        from .stanza_backend import StanzaBackend

        package = config.stanza_package or "perseus"
        use_gpu = not config.cpu_only
        backend = StanzaBackend(
            lang=config.language,
            package=package,
            use_gpu=use_gpu,
            processors="tokenize,mwt,pos,lemma",
        )
        return BuiltNLPBackend(
            backend=backend,
            info=NLPBackendInfo(
                name="stanza",
                language=config.language,
                package=package,
                use_gpu=use_gpu,
            ),
        )

    if config.backend == "transformers":
        if not config.model_name:
            raise NLPBackendConfigError(
                "nlp.model_name is required when nlp.backend=transformers"
            )
        from .transformers_backend import TransformersBackend

        use_gpu = not config.cpu_only
        backend = TransformersBackend(model_name=config.model_name)
        return BuiltNLPBackend(
            backend=backend,
            info=NLPBackendInfo(
                name="transformers",
                language=config.language,
                model=config.model_name,
                use_gpu=use_gpu,
            ),
        )

    raise NLPBackendConfigError("nlp.backend must be one of: stanza, transformers")


def render_backend_info(info: NLPBackendInfo) -> list[str]:
    lines = [
        f"backend={info.name}",
        f"language={info.language}",
        f"device={info.device}",
    ]
    if info.package is not None:
        lines.append(f"package={info.package}")
    if info.model is not None:
        lines.append(f"model={info.model}")
    return lines
