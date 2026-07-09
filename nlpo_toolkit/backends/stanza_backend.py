from ..models import NLPDocument, NLPSentence, NLPToken


class StanzaBackendUnavailableError(RuntimeError):
    pass


def convert_stanza_document_to_common_model(stanza_doc, original_text: str) -> NLPDocument:
    """Convert a Stanza document-like object into the common NLPDocument model."""
    doc = NLPDocument(text=original_text)

    sentences = getattr(stanza_doc, "sentences", [])

    for stanza_sent in sentences:
        sent_model = NLPSentence(text=getattr(stanza_sent, "text", None))

        words = list(getattr(stanza_sent, "words", []) or [])
        if not words and hasattr(stanza_sent, "tokens"):
            for token in stanza_sent.tokens:
                words.extend(getattr(token, "words", []))

        for word in words:
            sent_model.tokens.append(NLPToken(
                text=getattr(word, "text", ""),
                lemma=getattr(word, "lemma", None),
                upos=getattr(word, "upos", None),
                start_char=getattr(word, "start_char", None),
                end_char=getattr(word, "end_char", None),
            ))

        doc.sentences.append(sent_model)

    return doc


class StanzaBackend:
    def __init__(
        self,
        lang: str = "la",
        package: str = "perseus",
        use_gpu: bool = False,
        processors: str = "tokenize,mwt,pos,lemma"
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
