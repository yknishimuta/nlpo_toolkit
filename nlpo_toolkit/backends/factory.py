from __future__ import annotations

from nlpo_toolkit.nlp.contracts import (
    BuiltNLPBackend,
    NLPBackendInfo,
    NLPBackendSpec,
)


class NLPBackendConfigError(ValueError):
    pass


def create_nlp_backend(
    spec: NLPBackendSpec,
    *,
    processors: tuple[str, ...],
) -> BuiltNLPBackend:
    if spec.backend == "stanza":
        from .stanza_backend import StanzaBackend

        package = spec.stanza_package or "perseus"
        backend = StanzaBackend(
            lang=spec.language,
            package=package,
            use_gpu=spec.use_gpu,
            processors=",".join(processors),
        )
        return BuiltNLPBackend(
            backend=backend,
            info=NLPBackendInfo(
                name="stanza",
                language=spec.language,
                package=package,
                use_gpu=spec.use_gpu,
            ),
        )

    if spec.backend == "transformers":
        if not spec.model_name:
            raise NLPBackendConfigError(
                "nlp.model_name is required when nlp.backend=transformers"
            )
        from .transformers_backend import TransformersBackend

        backend = TransformersBackend(model_name=spec.model_name)
        return BuiltNLPBackend(
            backend=backend,
            info=NLPBackendInfo(
                name="transformers",
                language=spec.language,
                model=spec.model_name,
                use_gpu=spec.use_gpu,
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
