from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from nlpo_toolkit.immutable_collections import freeze_mapping
from nlpo_toolkit.stylometry.evaluation_models import (
    AuthorshipMetadata,
    LabeledFeatureDataset,
    LabeledFeatureObservation,
)

from ...nlp.roman_numerals import RomanExceptionsError
from ..config_references import ConfigReferenceError
from ..execution_session import prepare_analysis_corpus_session, start_nlp_execution_session
from ..ports import FeatureCommandDependencies
from .engine import analyze_feature_corpora
from .errors import FeatureError
from .function_words import build_function_word_columns
from .models import (
    AnalyzedFeatureCorpus,
    FeatureFilterPolicy,
    FeatureOptions,
    FeatureRequest,
    FeatureRow,
    FunctionWordOptions,
)


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


@dataclass(frozen=True)
class PreparedCorpusStylometryData:
    analyzed: tuple[AnalyzedFeatureCorpus, ...]
    options: FeatureOptions
    assignments: Mapping[str, tuple[str, str]]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "analyzed", tuple(self.analyzed))
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "assignments", freeze_mapping(self.assignments))


def validate_corpus_authorship_metadata(
    labels: tuple[str, ...], metadata: AuthorshipMetadata
) -> Mapping[str, tuple[str, str]]:
    assignments = {
        item.observation_id: (item.author, item.work_id)
        for item in metadata.assignments
    }
    label_set = set(labels)
    missing = tuple(label for label in labels if label not in assignments)
    unknown = tuple(identifier for identifier in assignments if identifier not in label_set)
    if missing:
        raise FeatureError(f"prepared corpus has no authorship metadata: {missing[0]!r}")
    if unknown:
        raise FeatureError(f"authorship metadata contains unknown group: {unknown[0]!r}")
    return freeze_mapping(assignments)


def prepare_corpus_stylometry_data(
    features: FeatureRequest,
    *,
    metadata: AuthorshipMetadata,
    dependencies: FeatureCommandDependencies,
) -> PreparedCorpusStylometryData:
    source = features.function_words
    function_words = None
    if source is not None:
        root = features.corpus.project_root.expanduser().resolve()
        path = source.path.expanduser()
        resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
        vocabulary = dependencies.load_function_words(resolved)
        build_function_word_columns(vocabulary)
        function_words = FunctionWordOptions(vocabulary, source.field)
    try:
        session = prepare_analysis_corpus_session(
            features.corpus, dependencies=dependencies.corpus
        )
    except (ConfigReferenceError, FileNotFoundError, RomanExceptionsError) as exc:
        raise FeatureError(str(exc)) from exc
    labels = tuple(corpus.label for corpus in session.corpora)
    assignments = validate_corpus_authorship_metadata(labels, metadata)
    if features.character_ngrams is not None:
        for corpus in session.corpora:
            if len(corpus.files) != 1:
                raise FeatureError(
                    "character n-gram features require one source file per prepared corpus; "
                    "use --group-by-file or grouping.mode: per_file"
                )
    nlp_session = start_nlp_execution_session(session, dependencies=dependencies.nlp)
    config = session.plan.definition.config
    options = FeatureOptions(
        field=features.field,
        mfw=features.mfw,
        include_upos=features.include_upos,
        include_basic=features.include_basic,
        filter_policy=FeatureFilterPolicy(
            min_token_length=config.filters.min_token_length,
            drop_roman_numerals=config.filters.drop_roman_numerals,
            roman_exceptions=nlp_session.roman_exceptions,
        ),
        sampling=features.sampling,
        lexical_diversity=features.lexical_diversity,
        function_words=function_words,
        character_ngrams=features.character_ngrams,
        upos_ngrams=features.upos_ngrams,
        morphology=features.morphology,
    )
    analyzed = analyze_feature_corpora(
        session.corpora,
        nlp=nlp_session.backend.backend,
        extraction_policy=nlp_session.extraction_policy,
        filter_policy=options.filter_policy,
    )
    return PreparedCorpusStylometryData(analyzed, options, assignments, labels)


def build_labeled_feature_dataset(
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
        try:
            author, work = assignments[group]
        except KeyError as exc:
            raise FeatureError(f"feature row has no authorship metadata: {group!r}") from exc
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
