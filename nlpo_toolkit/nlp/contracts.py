from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

__all__ = [
    "BuiltNLPBackend",
    "NLPBackend",
    "NLPBackendInfo",
    "NLPBackendSpec",
    "NLPDocument",
    "NLPSentence",
    "NLPToken",
]


@dataclass(frozen=True)
class NLPToken:
    text: str
    lemma: str | None
    upos: str | None
    start_char: int | None = None
    end_char: int | None = None


@dataclass(frozen=True)
class NLPSentence:
    tokens: tuple[NLPToken, ...] = ()
    text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", tuple(self.tokens))


@dataclass(frozen=True)
class NLPDocument:
    sentences: tuple[NLPSentence, ...] = ()
    text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sentences", tuple(self.sentences))


class NLPBackend(Protocol):
    def __call__(self, text: str) -> NLPDocument: ...


@dataclass(frozen=True)
class NLPBackendInfo:
    name: str
    language: str
    model: str | None = None
    package: str | Mapping[str, str] | None = None
    use_gpu: bool = False

    @property
    def device(self) -> str:
        return "gpu" if self.use_gpu else "cpu"

@dataclass(frozen=True)
class BuiltNLPBackend:
    backend: NLPBackend
    info: NLPBackendInfo


@dataclass(frozen=True)
class NLPBackendSpec:
    backend: str
    language: str
    stanza_package: str | None = None
    model_name: str | None = None
    use_gpu: bool = False
