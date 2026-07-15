from nlpo_toolkit.nlp.contracts import NLPDocument, NLPSentence, NLPToken


class StanzaBackendUnavailableError(RuntimeError):
    pass


def convert_stanza_document_to_common_model(stanza_doc, original_text: str) -> NLPDocument:
    """Convert a Stanza document-like object into the common NLPDocument model."""
    sentences: list[NLPSentence] = []
    for stanza_sent in getattr(stanza_doc, "sentences", []):
        words = list(getattr(stanza_sent, "words", []) or [])
        if not words and hasattr(stanza_sent, "tokens"):
            for token in stanza_sent.tokens:
                words.extend(getattr(token, "words", []))

        tokens = tuple(
            NLPToken(
                text=getattr(word, "text", ""),
                lemma=getattr(word, "lemma", None),
                upos=getattr(word, "upos", None),
                start_char=getattr(word, "start_char", None),
                end_char=getattr(word, "end_char", None),
            )
            for word in words
        )
        sentences.append(
            NLPSentence(tokens=tokens, text=getattr(stanza_sent, "text", None))
        )
    return NLPDocument(sentences=tuple(sentences), text=original_text)


class StanzaBackend:
    def __init__(
        self,
        lang: str = "la",
        package: str = "perseus",
        use_gpu: bool = False,
        *,
        processors: str,
    ):
        try:
            import stanza
        except ImportError as exc:
            raise StanzaBackendUnavailableError(
                "The stanza backend requires the optional 'stanza' dependency"
            ) from exc

        self.pipeline = stanza.Pipeline(
            lang=lang,
            processors=processors,
            package=package,
            use_gpu=use_gpu
        )

    def __call__(self, text: str) -> NLPDocument:
        """
        Takes text, parses it with Stanza, and returns it converted to a common data model.
        Implements __call__ so that callers can use it as a function, like nlp(text).
        """
        stanza_doc = self.pipeline(text)
        return convert_stanza_document_to_common_model(stanza_doc, text)

    def _convert_to_common_model(self, stanza_doc, original_text: str) -> NLPDocument:
        return convert_stanza_document_to_common_model(stanza_doc, original_text)
