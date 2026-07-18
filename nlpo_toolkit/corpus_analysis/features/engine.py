from __future__ import annotations

from nlpo_toolkit.nlp.contracts import NLPBackend

from ..analysis_policy import AnalysisExtractionPolicy
from ..analysis_records import iter_nlp_analysis_records_from_text
from ..corpus import PreparedCorpus
from .errors import FeatureError
from .filtering import filter_feature_records
from .lexical import compute_basic_features
from .mfw import compute_mfw_features, select_mfw_terms
from .models import (
    AnalyzedFeatureCorpus,
    FeatureFilterPolicy,
    FeatureOptions,
    FeatureRow,
    FeatureScalar,
    validate_feature_options,
)
from .sampling import sample_feature_corpus
from .upos import compute_upos_features


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
    )


def build_feature_matrix(
    *,
    corpora: tuple[PreparedCorpus, ...],
    nlp: NLPBackend,
    extraction_policy: AnalysisExtractionPolicy,
    options: FeatureOptions,
) -> tuple[FeatureRow, ...]:
    validate_feature_options(options)
    analyzed = tuple(
        analyze_feature_corpus(
            corpus,
            nlp=nlp,
            extraction_policy=extraction_policy,
            filter_policy=options.filter_policy,
        )
        for corpus in corpora
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
        if options.include_upos:
            values.update(compute_upos_features(corpus.lexical_records))
        if terms:
            values.update(
                compute_mfw_features(
                    corpus.lexical_records, terms=terms, field=options.field
                )
            )
        rows.append(FeatureRow.from_mapping(values))
    return tuple(rows)
