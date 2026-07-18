from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from nlpo_toolkit.stylometry.authorship import (
    build_work_profiles,
    validate_lowo_dataset,
)
from nlpo_toolkit.stylometry.evaluation import evaluate_lowo_fold
from nlpo_toolkit.stylometry.evaluation_models import (
    AuthorshipMetadata,
    LabeledFeatureDataset,
    LabeledFeatureObservation,
)
from nlpo_toolkit.stylometry.evaluation_results import (
    AuthorEvaluationSummary,
    LeaveOneWorkOutSummary,
)
from nlpo_toolkit.stylometry.ports import AuthorshipMetadataReader

from ...nlp.roman_numerals import RomanExceptionsError
from ..config_references import ConfigReferenceError
from ..execution_session import (
    prepare_analysis_corpus_session,
    start_nlp_execution_session,
)
from ..ports import FeatureCommandDependencies
from .corpus_lowo_models import (
    CorpusLowoFoldResult,
    CorpusLowoRequest,
    CorpusLowoResult,
    FoldVocabularyAudit,
)
from .engine import analyze_feature_corpora, build_feature_rows, fit_feature_vocabulary
from .errors import FeatureError
from .function_words import build_function_word_columns
from .models import FeatureFilterPolicy, FeatureOptions, FeatureRow, FunctionWordOptions


@dataclass(frozen=True)
class CorpusLowoDependencies:
    features: FeatureCommandDependencies
    read_metadata: AuthorshipMetadataReader


_ROW_METADATA = frozenset(
    {
        "group",
        "source_file",
        "sample_id",
        "sample_index",
        "sample_start_token",
        "sample_end_token",
        "sample_kind",
    }
)


def _validate_metadata(
    labels: tuple[str, ...], metadata: AuthorshipMetadata
) -> dict[str, tuple[str, str]]:
    assignments = {
        item.observation_id: (item.author, item.work_id)
        for item in metadata.assignments
    }
    label_set = set(labels)
    missing = tuple(label for label in labels if label not in assignments)
    unknown = tuple(
        identifier for identifier in assignments if identifier not in label_set
    )
    if missing:
        raise FeatureError(
            f"prepared corpus has no authorship metadata: {missing[0]!r}"
        )
    if unknown:
        raise FeatureError(
            f"authorship metadata contains unknown group: {unknown[0]!r}"
        )
    return assignments


def _labeled_rows(
    rows: tuple[FeatureRow, ...],
    assignments: Mapping[str, tuple[str, str]],
) -> LabeledFeatureDataset:
    if not rows:
        raise FeatureError("feature transformation produced no rows")
    feature_names = tuple(
        name
        for name, value in rows[0].items()
        if name not in _ROW_METADATA and isinstance(value, (int, float))
    )
    observations = []
    for row in rows:
        group = str(row["group"])
        author, work = assignments[group]
        identifier = str(row.values.get("sample_id", group))
        observations.append(
            LabeledFeatureObservation(
                identifier,
                author,
                work,
                tuple(float(row[name]) for name in feature_names),
            )
        )
    return LabeledFeatureDataset(feature_names, tuple(observations))


def execute_corpus_lowo(
    request: CorpusLowoRequest, *, dependencies: CorpusLowoDependencies
) -> CorpusLowoResult:
    source = request.features.function_words
    function_words = None
    if source is not None:
        root = request.corpus.project_root.expanduser().resolve()
        path = source.path.expanduser()
        resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
        vocabulary = dependencies.features.load_function_words(resolved)
        build_function_word_columns(vocabulary)
        function_words = FunctionWordOptions(vocabulary, source.field)
    metadata = dependencies.read_metadata(
        request.metadata_path,
        input_format=request.metadata_format,
        id_column=request.metadata_group_column,
        author_column=request.author_column,
        work_column=request.work_column,
    )
    try:
        session = prepare_analysis_corpus_session(
            request.corpus, dependencies=dependencies.features.corpus
        )
    except (ConfigReferenceError, FileNotFoundError, RomanExceptionsError) as exc:
        raise FeatureError(str(exc)) from exc
    labels = tuple(corpus.label for corpus in session.corpora)
    assignments = _validate_metadata(labels, metadata)
    if request.features.character_ngrams is not None:
        for corpus in session.corpora:
            if len(corpus.files) != 1:
                raise FeatureError(
                    "character n-gram features require one source file per prepared corpus; "
                    "use --group-by-file or grouping.mode: per_file"
                )
    nlp_session = start_nlp_execution_session(
        session, dependencies=dependencies.features.nlp
    )
    config = session.plan.definition.config
    options = FeatureOptions(
        field=request.features.field,
        mfw=request.features.mfw,
        include_upos=request.features.include_upos,
        include_basic=request.features.include_basic,
        filter_policy=FeatureFilterPolicy(
            min_token_length=config.filters.min_token_length,
            drop_roman_numerals=config.filters.drop_roman_numerals,
            roman_exceptions=nlp_session.roman_exceptions,
        ),
        sampling=request.features.sampling,
        lexical_diversity=request.features.lexical_diversity,
        function_words=function_words,
        character_ngrams=request.features.character_ngrams,
        upos_ngrams=request.features.upos_ngrams,
    )
    analyzed = analyze_feature_corpora(
        session.corpora,
        nlp=nlp_session.backend.backend,
        extraction_policy=nlp_session.extraction_policy,
        filter_policy=options.filter_policy,
    )
    work_order = []
    for label in labels:
        work = assignments[label][1]
        if work not in work_order:
            work_order.append(work)
    folds = []
    for fold_index, test_work in enumerate(work_order, start=1):
        training = tuple(
            corpus
            for corpus in analyzed
            if assignments[corpus.source.label][1] != test_work
        )
        testing = tuple(
            corpus
            for corpus in analyzed
            if assignments[corpus.source.label][1] == test_work
        )
        vocabulary = fit_feature_vocabulary(training, options=options)
        rows = build_feature_rows(
            training + testing, options=options, vocabulary=vocabulary
        )
        labeled = _labeled_rows(rows, assignments)
        validate_lowo_dataset(labeled)
        profiles = build_work_profiles(labeled)
        test_profile = next(
            profile for profile in profiles if profile.work_id == test_work
        )
        training_profiles = tuple(
            profile for profile in profiles if profile.work_id != test_work
        )
        evaluation = evaluate_lowo_fold(
            labeled.feature_names,
            training_work_profiles=training_profiles,
            test_work_profile=test_profile,
            fold_index=fold_index,
        )
        audit = FoldVocabularyAudit(
            fold_index,
            test_work,
            vocabulary.mfw_terms,
            vocabulary.character_ngrams.terms if vocabulary.character_ngrams else (),
            vocabulary.upos_ngrams.terms if vocabulary.upos_ngrams else (),
        )
        folds.append(
            CorpusLowoFoldResult(evaluation, len(labeled.feature_names), audit)
        )
    base_folds = tuple(item.evaluation for item in folds)
    author_order = []
    for fold in base_folds:
        if fold.actual_author not in author_order:
            author_order.append(fold.actual_author)
    per_author = tuple(
        AuthorEvaluationSummary(
            author,
            sum(fold.actual_author == author for fold in base_folds),
            sum(
                fold.actual_author == author and fold.is_correct for fold in base_folds
            ),
        )
        for author in author_order
    )
    summary = LeaveOneWorkOutSummary(per_author, base_folds)
    return CorpusLowoResult(tuple(folds), summary)
