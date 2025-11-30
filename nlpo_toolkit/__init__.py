from .nlp import (
    build_stanza_pipeline,
    tokenize_all_pos,
    count_nouns,
)

__all__ = [
    "build_stanza_pipeline",
    "tokenize_all_pos",
    "count_nouns",
]

__version__ = "0.1.1"