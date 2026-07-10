from __future__ import annotations

from typing import Any

from ..models import NLPDocument, NLPSentence, NLPToken


class NLPBackendUnavailableError(RuntimeError):
    pass


class TransformersBackend:
    """
    Minimal Hugging Face token-classification backend.

    This preserves the previous adapter behavior: the model output word is used
    as the lemma fallback, entity labels are mapped coarsely to UPOS, and no
    additional sentence segmentation or Latin-specific lemmatization is added.
    """

    def __init__(self, model_name: str):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise NLPBackendUnavailableError(
                "The transformers backend requires the optional 'transformers' dependency"
            ) from exc

        self.pos_pipeline = pipeline("token-classification", model=model_name)

    def __call__(self, text: str) -> NLPDocument:
        hf_outputs = self.pos_pipeline(text)
        tokens: list[NLPToken] = []
        for item in hf_outputs:
            word = str(item.get("word", ""))
            tokens.append(
                NLPToken(
                    text=word,
                    lemma=word.lower(),
                    upos=self._map_to_upos(str(item.get("entity", ""))),
                    start_char=_optional_int(item.get("start")),
                    end_char=_optional_int(item.get("end")),
                )
            )

        return NLPDocument(
            sentences=[NLPSentence(tokens=tokens, text=text)],
            text=text,
        )

    def _map_to_upos(self, entity_tag: str) -> str:
        if "NOUN" in entity_tag:
            return "NOUN"
        if "VERB" in entity_tag:
            return "VERB"
        return "X"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
