from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from nlpo_toolkit.nlp.contracts import NLPDocument, NLPSentence, NLPToken


class NLPBackendUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class TransformerTokenOutput:
    word: str
    entity: str
    start: int | None
    end: int | None


def _optional_position(raw: object, *, index: int, field: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise NLPBackendUnavailableError(
            f"Invalid transformers output {index} field {field}: expected integer or null"
        )
    return raw


def parse_transformer_token_output(raw: object, *, index: int) -> TransformerTokenOutput:
    if not isinstance(raw, Mapping):
        raise NLPBackendUnavailableError(
            f"Invalid transformers output {index}: expected mapping"
        )
    word = raw.get("word")
    entity = raw.get("entity")
    if not isinstance(word, str):
        raise NLPBackendUnavailableError(
            f"Invalid transformers output {index} field word: expected string"
        )
    if not isinstance(entity, str):
        raise NLPBackendUnavailableError(
            f"Invalid transformers output {index} field entity: expected string"
        )
    return TransformerTokenOutput(
        word, entity,
        _optional_position(raw.get("start"), index=index, field="start"),
        _optional_position(raw.get("end"), index=index, field="end"),
    )


def parse_transformer_outputs(raw: object) -> tuple[TransformerTokenOutput, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise NLPBackendUnavailableError("Invalid transformers output: expected sequence")
    return tuple(
        parse_transformer_token_output(item, index=index)
        for index, item in enumerate(raw)
    )


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
        hf_outputs = parse_transformer_outputs(self.pos_pipeline(text))
        tokens: list[NLPToken] = []
        for item in hf_outputs:
            word = item.word
            tokens.append(
                NLPToken(
                    text=word,
                    lemma=word.lower(),
                    upos=self._map_to_upos(item.entity),
                    start_char=item.start,
                    end_char=item.end,
                )
            )

        return NLPDocument(
            sentences=(NLPSentence(tokens=tuple(tokens), text=text),),
            text=text,
        )

    def _map_to_upos(self, entity_tag: str) -> str:
        if "NOUN" in entity_tag:
            return "NOUN"
        if "VERB" in entity_tag:
            return "VERB"
        return "X"
