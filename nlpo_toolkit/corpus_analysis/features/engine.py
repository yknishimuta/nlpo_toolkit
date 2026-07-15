from __future__ import annotations

from nlpo_toolkit.nlp.contracts import NLPBackend

from ..analysis_policy import AnalysisExtractionPolicy
from ..analysis_records import iter_nlp_analysis_records_from_text
from ..corpus import PreparedCorpus
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
    return AnalyzedFeatureCorpus(
        source=corpus,
        raw_record_count=len(raw_records),
        sentence_count=len(
            {(record.chunk_index, record.sentence_index) for record in raw_records}
        ),
        records=filter_feature_records(raw_records, policy=filter_policy),
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
    rows: list[FeatureRow] = []
    for corpus in analyzed:
        values: dict[str, FeatureScalar] = {"group": corpus.source.label}
        if options.include_basic:
            values.update(compute_basic_features(corpus))
        if options.include_upos:
            values.update(compute_upos_features(corpus.records))
        if terms:
            values.update(
                compute_mfw_features(corpus.records, terms=terms, field=options.field)
            )
        rows.append(FeatureRow.from_mapping(values))
    return tuple(rows)
