from __future__ import annotations

from nlpo_toolkit.nlp.roman_numerals import RomanExceptionsError

from ..config_references import ConfigReferenceError
from ..execution_session import (
    prepare_analysis_corpus_session,
    start_nlp_execution_session,
)
from ..ports import FeatureCommandDependencies
from .engine import build_feature_matrix, prepare_character_ngram_vocabulary
from .errors import FeatureError
from .function_words import build_function_word_columns
from .models import (
    FeatureCommandResult,
    FeatureFilterPolicy,
    FeatureOptions,
    FeatureRequest,
    FunctionWordOptions,
)


def execute_feature_command(
    request: FeatureRequest,
    *,
    dependencies: FeatureCommandDependencies,
) -> FeatureCommandResult:
    function_words = None
    if request.function_words is not None:
        source = request.function_words
        project_root = request.corpus.project_root.expanduser().resolve()
        path = source.path.expanduser()
        resolved_path = (
            (project_root / path).resolve()
            if not path.is_absolute()
            else path.resolve()
        )
        vocabulary = dependencies.load_function_words(resolved_path)
        build_function_word_columns(vocabulary)
        function_words = FunctionWordOptions(
            vocabulary=vocabulary,
            field=source.field,
        )
    try:
        corpus_session = prepare_analysis_corpus_session(
            request.corpus,
            dependencies=dependencies.corpus,
        )
        character_vocabulary = prepare_character_ngram_vocabulary(
            corpus_session.corpora,
            options=request.character_ngrams,
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
        sampling=request.sampling,
        lexical_diversity=request.lexical_diversity,
        function_words=function_words,
        character_ngrams=request.character_ngrams,
        upos_ngrams=request.upos_ngrams,
    )
    return FeatureCommandResult(
        rows=build_feature_matrix(
            corpora=corpus_session.corpora,
            nlp=nlp_session.backend.backend,
            extraction_policy=nlp_session.extraction_policy,
            options=options,
            character_vocabulary=character_vocabulary,
        )
    )
