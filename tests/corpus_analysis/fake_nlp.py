from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from nlpo_toolkit.backends import BuiltNLPBackend, NLPBackendInfo
from nlpo_toolkit.models import NLPDocument, NLPSentence, NLPToken


TokenSpec = tuple[str, str | None, str]


@dataclass
class FakeNLPBackend:
    tokens: Sequence[TokenSpec] | None = None
    per_text: dict[str, Sequence[TokenSpec]] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def __call__(self, text: str) -> NLPDocument:
        self.calls.append(text)
        specs = self.per_text.get(text, self.tokens)
        if specs is None:
            specs = tuple(
                (match.group(0), match.group(0).lower(), "NOUN")
                for match in re.finditer(r"\S+", text)
            )
        tokens: list[NLPToken] = []
        search_from = 0
        for surface, lemma, upos in specs:
            start = text.find(surface, search_from)
            if start < 0:
                start = None
                end = None
            else:
                end = start + len(surface)
                search_from = end
            tokens.append(
                NLPToken(
                    text=surface,
                    lemma=lemma,
                    upos=upos,
                    start_char=start,
                    end_char=end,
                )
            )
        return NLPDocument(
            sentences=[NLPSentence(tokens=tokens, text=text)],
            text=text,
        )


def fake_backend_factory(
    tokens: Iterable[TokenSpec] | None = None,
    *,
    backend: FakeNLPBackend | None = None,
):
    fake = backend or FakeNLPBackend(tokens=tuple(tokens) if tokens is not None else None)

    def factory(config):
        return BuiltNLPBackend(
            backend=fake,
            info=NLPBackendInfo(
                name="fake",
                language=getattr(config, "language", "la"),
                package="fake",
                use_gpu=False,
            ),
        )

    return factory
