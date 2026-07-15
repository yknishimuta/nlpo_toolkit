from __future__ import annotations

from nlpo_toolkit.nlp.roman_numerals import RomanExceptionsError

from ..config_references import ConfigReferenceError
from ..execution_session import (
    prepare_analysis_corpus_session,
    start_nlp_execution_session,
)
from ..ports import FeatureCommandDependencies
from .engine import build_feature_matrix
from .errors import FeatureError
from .models import FeatureCommandResult, FeatureFilterPolicy, FeatureOptions, FeatureRequest


def execute_feature_command(
    request: FeatureRequest,
    *,
    dependencies: FeatureCommandDependencies,
) -> FeatureCommandResult:
    try:
        corpus_session = prepare_analysis_corpus_session(
            request.corpus,
            dependencies=dependencies.corpus,
        )
        nlp_session = start_nlp_execution_session(
            corpus_session,
            dependencies=dependencies.nlp,
        )
    except (ConfigReferenceError, FileNotFoundError, RomanExceptionsError) as exc:
        raise FeatureError(str(exc)) from exc
    config = corpus_session.plan.definition.config
    options = FeatureOptions(
        field=request.field,
        mfw=request.mfw,
        include_upos=request.include_upos,
        include_basic=request.include_basic,
        filter_policy=FeatureFilterPolicy(
            min_token_length=config.filters.min_token_length,
            drop_roman_numerals=config.filters.drop_roman_numerals,
            roman_exceptions=nlp_session.roman_exceptions,
        ),
    )
    return FeatureCommandResult(
        rows=build_feature_matrix(
            corpora=corpus_session.corpora,
            nlp=nlp_session.backend.backend,
            extraction_policy=nlp_session.extraction_policy,
            options=options,
        )
    )
