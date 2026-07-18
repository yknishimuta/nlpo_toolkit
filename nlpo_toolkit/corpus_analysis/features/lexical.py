from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from ..analysis_records import NLPAnalysisRecord
from .descriptive import summarize_distribution
from .filtering import feature_token_value
from .models import AnalyzedFeatureCorpus, FeatureScalar


def build_sentence_lengths(
    *,
    raw_records: tuple[NLPAnalysisRecord, ...],
    lexical_records: tuple[NLPAnalysisRecord, ...],
) -> tuple[int, ...]:
    sentence_ids: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for record in raw_records:
        sentence_id = (record.chunk_index, record.sentence_index)
        if sentence_id not in seen:
            seen.add(sentence_id)
            sentence_ids.append(sentence_id)
    lexical_counts = Counter(
        (record.chunk_index, record.sentence_index) for record in lexical_records
    )
    return tuple(lexical_counts.get(sentence_id, 0) for sentence_id in sentence_ids)


def compute_basic_features(
    corpus: AnalyzedFeatureCorpus,
) -> Mapping[str, FeatureScalar]:
    raw_records = corpus.raw_records
    lexical_records = corpus.lexical_records
    sentence_lengths = build_sentence_lengths(
        raw_records=raw_records,
        lexical_records=lexical_records,
    )
    token_lengths = tuple(
        len(feature_token_value(record)) for record in lexical_records
    )
    sentence_summary = summarize_distribution(sentence_lengths)
    token_summary = summarize_distribution(token_lengths)
    word_token_count = len(lexical_records)
    lemmas = [
        (record.lemma or record.token).strip().lower() for record in lexical_records
    ]
    tokens = [record.token.strip().lower() for record in lexical_records]
    lemma_counts = Counter(lemmas)
    lemma_type_count = len(lemma_counts)
    token_type_count = len(set(tokens))
    hapax_lemma_count = sum(count == 1 for count in lemma_counts.values())
    mean_sentence_length = (
        sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
    )
    mean_token_length = (
        sum(token_lengths) / len(token_lengths) if token_lengths else 0.0
    )
    return {
        "group": corpus.source.label,
        "file_count": len(corpus.source.files),
        "char_count": (
            corpus.char_count
            if corpus.char_count is not None
            else len(corpus.source.prepared_text)
        ),
        "sentence_count": corpus.sentence_count,
        "token_count": len(raw_records),
        "word_token_count": word_token_count,
        "lemma_type_count": lemma_type_count,
        "token_type_count": token_type_count,
        "hapax_lemma_count": hapax_lemma_count,
        "hapax_lemma_ratio": hapax_lemma_count / lemma_type_count
        if lemma_type_count
        else 0.0,
        "mean_sentence_length": mean_sentence_length,
        "sentence_length_variance": sentence_summary.variance,
        "sentence_length_median": sentence_summary.median,
        "sentence_length_q25": sentence_summary.q25,
        "sentence_length_q75": sentence_summary.q75,
        "mean_token_length": mean_token_length,
        "token_length_variance": token_summary.variance,
        "token_length_median": token_summary.median,
        "token_length_q25": token_summary.q25,
        "token_length_q75": token_summary.q75,
        "type_token_ratio": token_type_count / word_token_count
        if word_token_count
        else 0.0,
        "lemma_type_token_ratio": lemma_type_count / word_token_count
        if word_token_count
        else 0.0,
    }
