from __future__ import annotations

from nlpo_toolkit.nlp.contracts import NLPBackend

from ..analysis_policy import AnalysisExtractionPolicy
from ..analysis_records import iter_nlp_analysis_records_from_text
from ..corpus import PreparedCorpus
from .errors import FeatureError
from .character_ngrams import (
    CharacterNgramVocabulary,
    compute_character_ngram_features,
    select_character_ngram_vocabulary,
)
from .character_text import feature_unit_character_text
from .filtering import filter_feature_records
from .function_words import compute_function_word_features
from .lexical import compute_basic_features
from .lexical_diversity import compute_lexical_diversity_features
from .mfw import compute_mfw_features, select_mfw_terms
from .models import (
    AnalyzedFeatureCorpus,
    FeatureFilterPolicy,
    FeatureOptions,
    FeatureRow,
    FeatureScalar,
    CharacterNgramOptions,
    validate_feature_options,
)
from .sampling import sample_feature_corpus
from .upos import compute_upos_features
from .upos_ngrams import (
    compute_upos_ngram_features,
    select_upos_ngram_vocabulary,
)


def analyze_feature_corpus(
    corpus: PreparedCorpus,
    *,
    nlp: NLPBackend,
    extraction_policy: AnalysisExtractionPolicy,
    filter_policy: FeatureFilterPolicy,
) -> AnalyzedFeatureCorpus:
    raw_records = tuple(
        iter_nlp_analysis_records_from_text(
            text=corpus.prepared_text,
            nlp=nlp,
            policy=extraction_policy,
        )
    )
    lexical_records = filter_feature_records(raw_records, policy=filter_policy)
    return AnalyzedFeatureCorpus(
        source=corpus,
        raw_records=raw_records,
        lexical_records=lexical_records,
        text=corpus.prepared_text,
    )


def prepare_character_ngram_vocabulary(
    corpora: tuple[PreparedCorpus, ...],
    *,
    options: CharacterNgramOptions | None,
) -> CharacterNgramVocabulary | None:
    if options is None:
        return None
    for corpus in corpora:
        if len(corpus.files) != 1:
            raise FeatureError(
                "character n-gram features require one source file per prepared corpus; "
                "use --group-by-file or grouping.mode: per_file"
            )
    return select_character_ngram_vocabulary(
        tuple(corpus.prepared_text for corpus in corpora), options=options
    )


def build_feature_matrix(
    *,
    corpora: tuple[PreparedCorpus, ...],
    nlp: NLPBackend,
    extraction_policy: AnalysisExtractionPolicy,
    options: FeatureOptions,
    character_vocabulary: CharacterNgramVocabulary | None = None,
) -> tuple[FeatureRow, ...]:
    validate_feature_options(options)
    if options.character_ngrams is not None and character_vocabulary is None:
        character_vocabulary = prepare_character_ngram_vocabulary(
            corpora, options=options.character_ngrams
        )
    analyzed = tuple(
        analyze_feature_corpus(
            corpus,
            nlp=nlp,
            extraction_policy=extraction_policy,
            filter_policy=options.filter_policy,
        )
        for corpus in corpora
    )
    upos_ngram_vocabulary = (
        select_upos_ngram_vocabulary(analyzed, options=options.upos_ngrams)
        if options.upos_ngrams is not None
        else None
    )
    terms = select_mfw_terms(analyzed, count=options.mfw, field=options.field)
    units = tuple(
        unit
        for corpus in analyzed
        for unit in sample_feature_corpus(corpus, options=options.sampling)
    )
    if options.sampling.enabled and not units:
        raise FeatureError(
            "fixed-token sampling produced no samples; "
            "reduce --window-tokens or use --include-partial-window"
        )
    rows: list[FeatureRow] = []
    for corpus in units:
        values: dict[str, FeatureScalar] = {"group": corpus.source.label}
        if corpus.sample is not None:
            values.update(
                {
                    "source_file": corpus.sample.source_file,
                    "sample_id": corpus.sample.sample_id,
                    "sample_index": corpus.sample.sample_index,
                    "sample_start_token": corpus.sample.start_token,
                    "sample_end_token": corpus.sample.end_token,
                    "sample_kind": corpus.sample.kind,
                }
            )
        if options.include_basic:
            values.update(compute_basic_features(corpus))
        if options.lexical_diversity is not None:
            values.update(
                compute_lexical_diversity_features(
                    corpus.lexical_records,
                    options=options.lexical_diversity,
                )
            )
        if options.include_upos:
            values.update(compute_upos_features(corpus.lexical_records))
        if upos_ngram_vocabulary is not None:
            values.update(
                compute_upos_ngram_features(
                    corpus.lexical_records,
                    vocabulary=upos_ngram_vocabulary,
                )
            )
        if options.function_words is not None:
            values.update(
                compute_function_word_features(
                    corpus.lexical_records,
                    options=options.function_words,
                )
            )
        if character_vocabulary is not None:
            values.update(
                compute_character_ngram_features(
                    feature_unit_character_text(corpus),
                    vocabulary=character_vocabulary,
                )
            )
        if terms:
            values.update(
                compute_mfw_features(
                    corpus.lexical_records, terms=terms, field=options.field
                )
            )
        rows.append(FeatureRow.from_mapping(values))
    return tuple(rows)
