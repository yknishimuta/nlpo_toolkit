from .nlp import (
    build_stanza_pipeline,
    tokenize_all_pos,
    count_nouns,
)
from .backends import create_nlp_backend
from .interfaces import NLPBackend

__all__ = [
    "build_stanza_pipeline",
    "create_nlp_backend",
    "tokenize_all_pos",
    "count_nouns",
    "NLPBackend",
]

__version__ = "0.1.1"
