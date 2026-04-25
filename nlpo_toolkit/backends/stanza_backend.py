from typing import Optional
import stanza
from ..models import NLPDocument, NLPSentence, NLPToken

class StanzaBackend:
    def __init__(
        self,
        lang: str = "la",
        package: str = "perseus",
        use_gpu: bool = False,
        processors: str = "tokenize,mwt,pos,lemma"
    ):
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
        return self._convert_to_common_model(stanza_doc, text)

    def _convert_to_common_model(self, stanza_doc, original_text: str) -> NLPDocument:
        doc = NLPDocument(text=original_text)

        sentences = getattr(stanza_doc, "sentences", [])

        for stanza_sent in sentences:
            sent_model = NLPSentence(text=getattr(stanza_sent, "text", None))

            words = getattr(stanza_sent, "words", [])
            if not words and hasattr(stanza_sent, "tokens"):
                for token in stanza_sent.tokens:
                    words.extend(getattr(token, "words", []))

            for word in words:
                sent_model.tokens.append(NLPToken(
                    text=word.text,
                    lemma=getattr(word, "lemma", word.text.lower()),
                    upos=getattr(word, "upos", "X"),
                    start_char=getattr(word, "start_char", 0)
                ))

            doc.sentences.append(sent_model)

        return doc