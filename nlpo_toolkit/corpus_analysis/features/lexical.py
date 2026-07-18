from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from .models import AnalyzedFeatureCorpus, FeatureScalar


def compute_basic_features(
    corpus: AnalyzedFeatureCorpus,
) -> Mapping[str, FeatureScalar]:
    records = corpus.records
    word_token_count = len(records)
    lemmas = [(record.lemma or record.token).strip().lower() for record in records]
    tokens = [record.token.strip().lower() for record in records]
    lemma_counts = Counter(lemmas)
    lemma_type_count = len(lemma_counts)
    token_type_count = len(set(tokens))
    hapax_lemma_count = sum(count == 1 for count in lemma_counts.values())
    token_lengths = [len(token) for token in tokens]
    return {
        "group": corpus.source.label,
        "file_count": len(corpus.source.files),
        "char_count": (
            corpus.char_count
            if corpus.char_count is not None
            else len(corpus.source.prepared_text)
        ),
        "sentence_count": corpus.sentence_count,
        "token_count": corpus.raw_record_count,
        "word_token_count": word_token_count,
        "lemma_type_count": lemma_type_count,
        "token_type_count": token_type_count,
        "hapax_lemma_count": hapax_lemma_count,
        "hapax_lemma_ratio": hapax_lemma_count / lemma_type_count if lemma_type_count else 0.0,
        "mean_sentence_length": word_token_count / corpus.sentence_count if corpus.sentence_count else 0.0,
        "mean_token_length": sum(token_lengths) / len(token_lengths) if token_lengths else 0.0,
        "type_token_ratio": token_type_count / word_token_count if word_token_count else 0.0,
        "lemma_type_token_ratio": lemma_type_count / word_token_count if word_token_count else 0.0,
    }
