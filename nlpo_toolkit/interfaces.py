from typing import Protocol
from .models import NLPDocument

class NLPBackend(Protocol):
    """
    Protocol defining the common interface for NLP engines.
    Any backend (e.g., Stanza, Transformers) satisfying this interface
    can be transparently passed to the core logic.
    """

    def __call__(self, text: str) -> NLPDocument:
        """
        Receives raw text, performs morphological and/or syntactic analysis,
        and returns the result converted into a library-independent
        common data model (NLPDocument).

        Args:
            text (str): The plain text to be analyzed.

        Returns:
            NLPDocument: The common data model containing the analysis results.
        """
        ...