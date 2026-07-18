from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Protocol

from nlpo_toolkit.immutable_collections import freeze_mapping

__all__ = [
    "BuiltNLPBackend",
    "NLPBackend",
    "NLPBackendInfo",
    "NLPBackendSpec",
    "NLPDocument",
    "NLPSentence",
    "NLPToken",
    "UDMorphFeature",
]


@dataclass(frozen=True, order=True)
class UDMorphFeature:
    attribute: str
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.attribute, str) or not isinstance(self.value, str):
            raise TypeError("UD morphology attribute and value must be strings")
        attribute = self.attribute.strip()
        value = self.value.strip()
        if not attribute or not value:
            raise ValueError("UD morphology attribute and value must not be empty")
        if "=" in attribute or "|" in attribute or "|" in value:
            raise ValueError("invalid character in UD morphology feature")
        object.__setattr__(self, "attribute", attribute)
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class NLPToken:
    text: str
    lemma: str | None
    upos: str | None
    start_char: int | None = None
    end_char: int | None = None
    morphology: tuple[UDMorphFeature, ...] = ()

    def __post_init__(self) -> None:
        morphology = tuple(self.morphology)
        if any(not isinstance(item, UDMorphFeature) for item in morphology):
            raise TypeError("NLPToken morphology must contain UDMorphFeature values")
        if len({item.attribute for item in morphology}) != len(morphology):
            raise ValueError("NLPToken morphology attributes must be unique")
        object.__setattr__(self, "morphology", tuple(sorted(morphology)))


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

    def __post_init__(self) -> None:
        if isinstance(self.package, Mapping):
            object.__setattr__(self, "package", freeze_mapping(self.package))

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
